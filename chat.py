import pymysql.cursors
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import json

from pydub import AudioSegment
import simpleaudio as sa

from openai import OpenAI   
import pymysql
from datetime import datetime, date
# import openai      


# API 키 업로드 시 제거하기 !!
API_KEY = ''


client = OpenAI(api_key=API_KEY)    # API 키 지정


# 사용자가 3초 이상 말하지 않으면 녹음을 종료하고 파일에 저장하는 함수
def record_audio_with_silence_detection(filename, sample_rate=44100, silence_threshold=0.01, max_silence_duration=3):
    """
    Args:
        filename (str): 저장할 파일 이름 (예: 'output.wav')
        sample_rate (int): 샘플 레이트 (기본값은 44100Hz)
        silence_threshold (float): 무음 감지 임계값 (0 ~ 1 범위, 낮을수록 민감)
        max_silence_duration (int): 최대 무음 지속 시간 (초)
    """

    print("상담 진행 중입니다. 3초 이상 무음 시 녹음이 종료됩니다...")
    duration = 20  # 최대 녹음 시간 (초)
    recording = []
    silence_start_time = None

    # 스트림을 열어 실시간 녹음
    with sd.InputStream(samplerate=sample_rate, channels=1) as stream:
        start_time = time.time()
        while time.time() - start_time < duration:
            audio_chunk, _ = stream.read(int(sample_rate * 0.1))  # 0.1초 단위로 데이터 읽기
            recording.append(audio_chunk)
            
            # 오디오 데이터의 RMS 계산 (소리의 세기)
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            
            # 무음 구간을 감지
            if rms < silence_threshold:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time >= max_silence_duration:
                    break # 3초 이상 무음이 감지되어 녹음을 종료
            else:
                silence_start_time = None  # 소리가 있으면 무음 타이머 초기화

    # 녹음 데이터 저장
    recording = np.concatenate(recording, axis=0)
    write(filename, sample_rate, recording)
    print(f"{filename}에 녹음이 완료되었습니다.")


def speech_to_text(audio_path):
    stt_model = "whisper-1"                       # 모델명 지정
    audio_file = open(audio_path, "rb")

    transcript = client.audio.transcriptions.create(
    model=stt_model,
    language = "ko" ,
    file=audio_file,
    response_format = "text" , #json, text, srt, verbose_json, vtt.
    temperature = 0.5
    )
    return transcript

def text_to_speech(assistant_content):
    tts_model = "tts-1"                         # 모델명 지정
    audio_path = "assistant.mp3"

    response = client.audio.speech.create(
    model=tts_model,
    voice="fable",                              #alloy, echo, fable, onyx, nova, shimmer
    response_format = "mp3" , #mp3, opus, aac, flac.
    speed = 1.0, # 0.25 ~ 4
    input=assistant_content
    )

    response.stream_to_file(audio_path)         # 응답을 파일로 저장


    audio = AudioSegment.from_mp3(audio_path)   # MP3 파일 로드
    
                                                # AudioSegment 데이터를 배열로 변환하여 simpleaudio로 재생
    playback = sa.play_buffer(
        audio.raw_data,
        num_channels=audio.channels,
        bytes_per_sample=audio.sample_width,
        sample_rate=audio.frame_rate
    )
    playback.wait_done()                        # 재생이 끝날 때까지 대기

functions = [
    {
        'name':'get_user_info',
        'description': '사용자의 정보를 알아봅니다',
        'parameters':{
            'type':'object',
            'properties':{
                'name':{
                    'type':'string',
                    'description':'사용자의 이름입니다'
                }
            },
            'required':['name']
        },
    },
    {
        'name': 'insert_user_info',
        'description': '사용자의 정보를 SQL 데이터베이스에 등록하는 함수',
        'parameters': {
            'type': 'object',
            'properties':{
                'name': {'type': 'string', 'description': '사용자의 이름'},
                'age': {'type': 'integer', 'description': '사용자의 나이'},
                'date_of_birth': {'type': 'string', 'format': 'date', 'description': '사용자의 생일 (YYYY-MM-DD 형식'},
                'gender': {'type': 'string', 'description': '사용자의 성별'},
            }
        },
        'required': ['name', 'age']  
    },
        {
        'name': 'update_user_info',
        'description': '사용자의 정보를 SQL 데이터베이스에서 갱신하는 함수',
        'parameters': {
            'type': 'object',
            'properties':{
                'name': {'type': 'string', 'description': '사용자의 이름'},
                'age': {'type': 'integer', 'description': '사용자의 나이'},
                'date_of_birth': {'type': 'string', 'format': 'date', 'description': '사용자의 생일 (YYYY-MM-DD 형식'},
                'gender': {'type': 'string', 'description': '사용자의 성별'},
                
            }
        },
        'required': ['name', 'age', 'date_of_birth', 'gender']  
    },
            {
        'name': 'insert_counseling_record',
        'description': '사용자의 상담 정보를 SQL 데이터베이스에서 등록하는 함수',
        'parameters': {
            'type': 'object',
            'properties':{
                'name': {'type': 'string', 'description': '사용자의 이름'},
                'session_content': {'type': 'integer', 'description': '사용자의 상담내용'},
                'counselor_name': {'type': 'string', 'description': '상담사 이름, 기본 값은 ChatGPT'},
                
            }
        },
        'required': ['name', 'age', 'date_of_birth', 'gender']  
    }
]

#################################################################
# 유저정보에 데이터 삽입 함수
def insert_user_info(name='default', age='default', date_of_birth='default', gender='default'):
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='1541',
        database='chat_care'
    )
    cursor = conn.cursor()
    
    # gender 값을 SQL 형식에 맞게 변환 (필요시)
    if gender == "남자":
        gender = "M"
    elif gender == "여자":
        gender = "F"
    
    # 'default'가 아닌 파라미터만 필터링
    params = { 
        'name': name,
        'age': age,
        'date_of_birth': date_of_birth,
        'gender': gender
    }
    filtered_params = {k: v for k, v in params.items() if v != 'default'}
    
    # 컬럼과 값 리스트 생성
    columns = ', '.join(filtered_params.keys())
    placeholders = ', '.join(['%s'] * len(filtered_params))
    values = tuple(filtered_params.values())
    
    # 동적 INSERT 쿼리 실행
    sql = f"INSERT INTO user_info ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, values)

    conn.commit()
    conn.close()
    return "사용자 정보가 데이터베이스에 성공적으로 등록되었습니다."

def update_user_info(name='default', age='default', date_of_birth='default', gender='default'):
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='1541',
        database='chat_care'
    )
    cursor = conn.cursor()

    if gender == "남자":
        gender = "M"
    elif gender == "여자":
        gender = "F"

    params = { 
        'name': name,
        'age': age,
        'date_of_birth': date_of_birth,
        'gender': gender
    }
    updates = [(key, value) for key, value in params.items() if value != 'default']

    # 각 파라미터에 대해 업데이트 실행
    for n, v in updates:
        cursor.execute(f"UPDATE user_info SET {n} = %s WHERE name = %s", (v, params['name']))

    conn.commit()
    conn.close()
    return "사용자 정보가 성공적으로 갱신 되었습니다."

def get_user_info(name):
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='1541',
        database='chat_care'
    )
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM user_info WHERE name = (%s)", (name, ))
    user_data = cursor.fetchone()
    conn.close()

    # # datetime 객체를 문자열로 변환
    # user_data_serializable = {
    #     key: (value.isoformat() if isinstance(value, (datetime, date)) else value)
    #     for key, value in user_data.items()
    # }

    return_msg = f'''아이디가 {user_data['user_id']}인 {user_data['name']}님의 나이는 {user_data['age']}살이며 생일은 
    {user_data['date_of_birth']}이고 성별은 {user_data['gender']}입니다. 
    정보등록일은 {user_data['created_at']}입니다.'''
    return return_msg

#################################################################
def insert_counseling_record(name, session_content, counselor_name=None):
    # session_content가 JSON 형식의 문자열이라면 리스트로 변환
    if isinstance(session_content, str):
        session_content = json.loads(session_content)

    content_only = " ".join(
        [f"사용자: {entry['content']}" if entry["role"] == "user" else f"상담사: {entry['content']}"
         for entry in session_content]
    )
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='1541',
        database='chat_care'
    )
    cursor = conn.cursor()

    # 현재 날짜와 시간을 session_date로 설정
    session_date = datetime.now()
    
    # SQL INSERT 쿼리 작성
    sql = """
        INSERT INTO counseling_records (name, session_date, session_content, counselor_name)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(sql, (name, session_date, content_only, counselor_name))
    
    conn.commit()
    conn.close()
    
    return "상담 기록이 성공적으로 저장되었습니다."
#################################################################

def check(content, message, total_content):
    function_name = message.function_call.name
    arguments = message.function_call.arguments  # 'get' 대신 직접 접근
    args = json.loads(arguments)

    name = args.get("name")
    age = args.get("age")
    date_of_birth = args.get("date_of_birth")
    gender = args.get("gender")

    params = {
        'name': name if name is not None else 'default',
        'age': age if age is not None else 'default',
        'date_of_birth': date_of_birth if date_of_birth is not None else 'default',
        'gender': gender if gender is not None else 'default'
}

    # 함수 호출에 따라 적절한 기능 수행
    if function_name == "get_user_info":
        if name is None:
            return "사용자 이름이 제공되지 않았습니다."
        function_response = get_user_info(name=name)

    elif function_name == "insert_user_info":
        function_response = insert_user_info(**params)

    elif function_name == 'update_user_info':
        function_response = update_user_info(**params)
    elif function_name == 'insert_counseling_record':

        print(message)
        function_response = insert_counseling_record(name, total_content, None)
    else:
        function_response = json.dumps({"error": "Invalid function call"})

    second_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": content},
            message,
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            },
        ],
    )
    return second_response.choices[0].message.content

def ask(message):
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages=message, 
        functions=functions, 
        function_call="auto"
        )
    # print(completion)
    return completion.choices[0].message

if __name__=="__main__":

    gpt_model = 'gpt-3.5-turbo'
    message = [{"role":"system", "content":"당신은 친절한 심리 상담사 입니다."}]

    print('상담 모드를 선택해주세요. (1. 채팅, 2. 음성, 3. 종료)')
    mode = int(input("모드 선택: "))

    while True:
        if mode == 1:
            user_content = input('사용지: ')
            if '상담 종료' in user_content:
                break
            message.append({"role":"user", "content":f'{user_content}'})
            
            msg = ask(message)

            assistant_content = ''
            if msg.function_call:
                assistant_content = check(user_content, msg, message)
            else:
                assistant_content = msg.content.strip()              # GPT 답변
            
            message.append({"role":"assistant", "content":f'{assistant_content}'})
            print(f'상담사: {assistant_content}')

        elif mode == 2:
            print('상담 내용을 말씀하세요.')
            audio_path = 'output.wav'
            record_audio_with_silence_detection('output.wav')
            user_content = speech_to_text(audio_path)

            if '상담 종료' in user_content:
                break

            message.append({"role":"user", "content":f'{user_content}'})

            msg = ask(message)

            assistant_content = ''
            if msg.function_call:
                assistant_content = check(user_content, msg)
            else:
                assistant_content = msg.content.strip()              # GPT 답변
            
            message.append({"role":"assistant", "content":f'{assistant_content}'})

            text_to_speech(assistant_content)

        elif mode == 3:
            break

#     record_audio_with_silence_detection('output.wav')
