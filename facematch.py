import face_recognition


fernanda_image = face_recognition.load_image_file('./img/known/fernanda_montenegro.jpg')
# face_encodings encode an image at 128 dimentions
fernanda_face_encoding = face_recognition.face_encodings(fernanda_image)[0]

unknown_image = face_recognition.load_image_file('./img/unknown/fernanda-600x398.jpg')

unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [fernanda_face_encoding], unknown_face_encoding)

if results[0]:
    print('Essa é a Fernanda Montenegro')
else:
    print('Essa NÃO é a Fernanda Montenegro')
