import face_recognition


fernanda_image = face_recognition.load_image_file('./img/known/faustao.jpg')
# face_encodings encode an image at 128 dimentions
fernanda_face_encoding = face_recognition.face_encodings(fernanda_image)[0]

unknown_image = face_recognition.load_image_file('./img/unknown/15843071585e6e9bd65826e_1584307158_3x2_lg.jpg')

unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [fernanda_face_encoding], unknown_face_encoding)

if results[0]:
    print('Essa é o Fausto')
else:
    print('Essa NÃO é o Fausto')
