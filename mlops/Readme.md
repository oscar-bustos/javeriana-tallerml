# Ejercicio de Laboratorio: De los Datos al Chatbot (Ciclo Completo de MLOps)

¡Bienvenidos al taller! Hoy dejaremos los modelos en libretas de código y los llevaremos al mundo real. Durante las próximas 3 horas, entrenarán múltiples modelos predictivos, empaquetarán el mejor en un contenedor Docker y lo conectarán a un bot de Telegram usando automatización sin código.



## 🎯 Objetivos del Taller

* **Entrenar y registrar** modelos de Machine Learning usando MLflow en Databricks.
* **Desplegar** un modelo predictivo como una API REST usando Flask y Docker en Hugging Face.
* **Consumir** la API a través de un chatbot en Telegram automatizado con n8n.

## 🛠️ Requisitos Previos

Antes de comenzar, asegúrate de tener:

* Cuenta activa en **Databricks Community Edition**.
* Cuenta activa en **Hugging Face**.
* **Postman** instalado en tu computadora o acceso a la versión web.
* Cuenta de **Telegram** (aplicación de escritorio o en tu celular).
* Cuenta de prueba gratuita (15 días) en **n8n Cloud** (puedes crearla en [n8n.io/cloud](https://n8n.io/cloud/)).
* Archivos base del taller descargados desde: [Repositorio de GitHub](https://github.com/tu-universidad/taller-mlops-base)

---

## Parte 1: Entrenamiento y Selección del Modelo con Databricks

En esta etapa, usaremos el dataset de *California Housing* para predecir el precio de las viviendas. Entrenaremos varios modelos para ver cuál tiene el mejor rendimiento.

### Paso 1: Configurar el entorno
1. Ingresa a Databricks y crea un nuevo clúster de computación.
2. Ve a la sección **Workspace**, haz clic derecho e importa el archivo `entrenamiento_base.ipynb` que descargaste del repositorio.
3. Asigna el cuaderno a tu clúster recién creado.

### Paso 2: Entrenamiento y MLflow
1. Ejecuta las celdas de la sección **"Preparación de Datos"**.
2. Ejecuta la celda **"Entrenamiento Manual"**. Este script probará automáticamente algoritmos como Regresión Lineal, Random Forest y Gradient Boosted Trees.
3. Haz clic en el ícono de **Experimentos** (o en el enlace de MLflow que aparece en la salida de la celda) para ver la interfaz de seguimiento.

> **Nota:** MLflow registra los parámetros, métricas (como el RMSE) y el modelo resultante de cada ejecución. Compara cuál algoritmo obtuvo el error más bajo.

### Paso 3: AutoML y Descarga
1. Ejecuta la celda de **"AutoML"**. Databricks buscará automáticamente los mejores hiperparámetros. Compara si superó tu mejor modelo manual.
2. Ejecuta la última celda del cuaderno para descargar el mejor modelo. Se guardará en tu computadora como un archivo `mejor_modelo.pkl`.

---

## Parte 2: Sirviendo el Modelo con Hugging Face y Docker

Ahora que tenemos nuestro "cerebro" (`.pkl`), necesitamos construirle un cuerpo (API) para que el mundo exterior pueda hacerle preguntas.

### Paso 1: Crear el "Space" en Hugging Face
1. Ve a tu perfil de Hugging Face y haz clic en **New Space**.
2. Dale un nombre (ej. `api-viviendas-tu-apellido`).
3. En la sección *Space SDK*, selecciona **Docker** y luego **Blank**.
4. Haz clic en **Create Space**.

### Paso 2: Subir archivos y desplegar
1. En la pestaña **Files** de tu nuevo Space, haz clic en **Add file** > **Upload files**.
2. Sube los siguientes archivos:
   * Tu modelo `mejor_modelo.pkl`.
   * El archivo `app.py` (código de la API de Flask).
   * El archivo `requirements.txt` (dependencias de Python).
   * El archivo `Dockerfile` (instrucciones para empaquetar todo).
3. Haz clic en **Commit changes**. Hugging Face comenzará a construir tu contenedor automáticamente. Espera hasta que el estado cambie de *Building* a *Running*.

### Paso 3: Probar la API con Postman
1. Abre Postman y crea una nueva petición **POST**.
2. Pega la URL directa de tu API: `https://tu-usuario-api-viviendas.hf.space/predict`
3. Ve a la pestaña **Body**, selecciona **raw** y cambia el formato de *Text* a **JSON**.
4. Pega el siguiente JSON de prueba y haz clic en **Send**:

```json
{
    "MedInc": 8.32,
    "HouseAge": 41.0,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322.0,
    "AveOccup": 2.55,
    "Latitude": 37.88,
    "Longitude": -122.23
}
```

> **Éxito:** Deberías recibir una respuesta en la parte inferior con la predicción del precio de la vivienda. ¡Tu modelo ya está en internet!

---

## Parte 3: El Chatbot Inteligente con Telegram y n8n

Nadie quiere abrir Postman para saber el precio de una casa, y mucho menos escribir código para preguntarlo. Vamos a conectar nuestra API a un bot de Telegram. Para que el bot entienda mensajes naturales como *"Tengo una casa de 10 años con 3 cuartos..."*, usaremos Inteligencia Artificial en n8n para extraer los datos automáticamente.

### Paso 1: Crear el Bot en Telegram
1. Abre Telegram y busca al usuario **@BotFather** (el verificador oficial de bots).
2. Envíale el comando `/newbot` y sigue las instrucciones para darle un nombre y un usuario (debe terminar en *bot*).
3. BotFather te entregará un **Token HTTP API** (una cadena larga de texto). Cópialo y mantenlo a la mano.

### Paso 2: Construir el "Cerebro" Conversacional en n8n
Ingresa a tu cuenta de n8n y haz clic en **Add Workflow**. Deberás configurar los siguientes nodos en orden:

* **Nodo 1 - El Oído (Telegram Trigger):**
  * Agrega un nodo **Telegram Trigger**.
  * Crea una nueva credencial y pega el Token de BotFather.
  * Configúralo para que escuche eventos de tipo `Message`.

* **Nodo 2 - El Traductor (OpenAI / AI Node):**
  * Agrega un nodo **OpenAI** (o el proveedor de IA que te asigne el profesor, como Groq o Anthropic).
  * **Acción:** Chat / Generate Text.
  * **System Prompt (El contexto):** Escribe lo siguiente para forzar a la IA a estructurar los datos:
    > "Eres un asistente de bienes raíces en California. El usuario te describirá una casa. Tu único trabajo es extraer los datos y responder ÚNICAMENTE con un JSON válido usando estas claves exactas: `MedInc` (Ingreso medio del barrio, ej: 8.32), `HouseAge` (Edad de la casa), `AveRooms` (Promedio de cuartos), `AveBedrms` (Promedio de baños/alcobas, ej: 1.0), `Population` (Población del barrio, ej: 300), `AveOccup` (Ocupantes promedio, ej: 2.5), `Latitude` (ej: 37.88), `Longitude` (ej: -122.23). Si el usuario no menciona algún dato, inventa un valor promedio razonable para California. NO respondas nada más que el JSON."
  * **User Message:** Conéctalo dinámicamente al texto que llegó del nodo de Telegram (ej. `{{ $json.message.text }}`).

* **Nodo 3 - El Puente (HTTP Request):**
  * Conecta este nodo después de la IA.
  * **Method:** `POST`
  * **URL:** La URL de tu API en Hugging Face (`https://tu-usuario-api-viviendas.hf.space/predict`).
  * **Body Parameters:** Selecciona enviar JSON (Raw/Custom) y pásale dinámicamente la respuesta que generó el nodo de IA (`{{ $json.message.content }}`).

* **Nodo 4 - La Voz (Telegram):**
  * Agrega un nodo regular de **Telegram**.
  * **Operación:** Send Message.
  * **Chat ID:** Mapea el ID del chat del primer nodo (`{{ $('Telegram Trigger').item.json.message.chat.id }}`).
  * **Text:** Escribe una respuesta amigable mezclando texto y el resultado de la API. Ejemplo:
    > "¡Hola! Analicé las características de la casa que me mencionaste. 🏠 Según mi modelo de Inteligencia Artificial, el precio estimado en el mercado es de ${{ $('HTTP Request').item.json.precio_estimado_miles_usd }} mil dólares."

### Paso 3: La Prueba de Fuego
1. En n8n, guarda tu workflow y haz clic en **Test Workflow** (o activa el switch de *Active*).
2. Abre Telegram, busca tu bot y escríbele un mensaje completamente natural:
   > *"Hola, quiero evaluar una casa. Tiene unos 15 años de antigüedad, tiene 5 cuartos en total y 2 baños. Vivimos 4 personas ahí. Está ubicada en la latitud 35.0 y longitud -120.0."*
3. Observa cómo fluye la información en n8n: La IA convierte tu texto en JSON, se lo envía a tu contenedor en Hugging Face, Hugging Face usa tu modelo de Databricks, y Telegram te devuelve el precio. 

**¡Acabas de construir un sistema de Machine Learning de extremo a extremo!**
