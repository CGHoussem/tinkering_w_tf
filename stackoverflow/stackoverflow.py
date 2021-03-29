import os
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import losses

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# download stackoverflow dataset
dataset_dir = "stack_overflow_16k"
if not os.path.exists("stack_overflow_16k"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
    dataset = tf.keras.utils.get_file(
        "stack_overflow_16k.tar.gz", url, untar=True, cache_dir=".", cache_subdir=""
    )
    os.makedirs(dataset_dir)
    shutil.move("test", os.path.join(dataset_dir, "test"))
    shutil.move("train", os.path.join(dataset_dir, "train"))
    dataset_dir = os.path.join(os.path.dirname(dataset), "stack_overflow_16k")

# get train & test directories
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# création de l'ensemble de validation (25% from each class)
batch_size = 32
seed = 42

raw_train_ds = preprocessing.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2, subset="training", seed=seed
)

raw_val_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed,
)

raw_test_ds = preprocessing.text_dataset_from_directory(test_dir, batch_size=batch_size)

max_features = 500
embedding_dim = 128
sequence_length = 500

vectorize_layer = TextVectorization(
    max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length
)

text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print('Building and compiling model..')
model = tf.keras.Sequential(
    [
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4),
    ]
)

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)
print('Compilating ended.')

print('Training model..')
history = model.fit(train_ds, validation_data=val_ds, epochs=40)
print('Trainind ended.')

#######################
history_dict = history.history
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("model_loss.png")

plt.clf()
plt.plot(epochs, acc, "ro", label="Training acc")
plt.plot(epochs, val_acc, "r", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("model_accuracy.png")
#######################

loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)
print("-" * 10)

# creating a new general model
print('Building and compiling export_model')
export_model = tf.keras.Sequential(
    [vectorize_layer, model, layers.Activation("sigmoid")]
)

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=["accuracy"],
)
print('Compilation ended.')

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

examples = [
    "how to pause loop while time is ticking i have made a timer where i can set time i want to wait and then do something..so this is my short timer func:..private void sleep(int interval, action action).{.    system.windows.forms.timer mytimer = new system.windows.forms.timer();.    mytimer.interval = interval; //interval is in ms   .    mytimer.start();.    mytimer.tick += (s, e) =&gt;.    {.        action();.        mytimer.stop();.    };.}...and im using this timer in loop:..foreach (string word in words).{.   sleep(5000, ()=&gt; myaction());                                           .}...without loop timer is great but in loop it wont work because loop wont stop and wait for those 5secs. it do all stuff imediately and starts timer again and again too fast...so what im trying to find out is how to make my loop wait until time runs out and myaction() is executed. im working on forms application so all threadin sleeps wont work here. also i tried all other timers but they used too much cpu.",
    'how to download .msi file in blank i want to download .msi file using blank.  i have tried to download file using following code..printwriter out = null;.fileinputstream filetodownload = null;.bufferedreader bufferedreader = null;.try {.        out = response.getwriter();.        filetodownload = new fileinputstream(download_directory + file_name);.        bufferedreader = new bufferedreader(new inputstreamreader(filetodownload));..        //response.setcontenttype(""application/text"");.        //response.setcontenttype(""application/x-msi"");.        //response.setcontenttype(""application/msi"");.        //response.setcontenttype(""octet-stream"");.        response.setcontenttype(""application/octet-stream"");.        //response.setcontenttype(""application/x-7z-compressed"");.        //response.setcontenttype(""application/zip"");.        response.setheader(""content-disposition"",""attachment; filename="" +file_name );.        response.setcontentlength(filetodownload.available());..        system.out.println(""n now file download is starting"");.        string nextline = """";.        while((nextline = bufferedreader.readline()) != null){.            out.println(nextline);.        }.        out.flush();                                ..    } catch (ioexception e) {.        out.write(""&lt;center&gt;&lt;h2&gt;the installer is not available on server&lt;/h2&gt;&lt;/center&gt;"");.        system.out.println(""n got exception while getting the input stream from the file==&gt;""+e);.        log.error(""error::"", e);.    }.    finally{.        if(null != bufferedreader){.            try {.                bufferedreader.close();.            } catch (ioexception e) {.                system.out.println(""n error in closing buffer reader==&gt;""+e);.                log.error(""error::"", e);.            }.        }// end of if..        if(null != filetodownload){.            try {.                filetodownload.close();.            } catch (ioexception e) {.                system.out.println(""n error in closing input stream==&gt;""+e);.                log.error(""error::"", e);.            }.        }// end of if.    }// end of finally',
    "is it legal to define two methods with the same name but different returning types? i've written a piece of code to determine a typical palindrome string. i did this by the definition of a reverse() method returning a string. i also eager to have the same method, but in the void form, because of some future needs..as i add the latter to the code, the valid output will become invalid..so, the question is that is it legal to define two methods with the same name but different returning types?.if not, please let me know how to write this code with the void-type method...class detector(object):.    def __init__(self,string):.        self.string = string..    forbidden = (' ','!','?','.','-','_','&amp;','%',\"#\",\",\")..    def eliminator(self):.        for item in self.forbidden:.            if item in self.string:.                self.string = self.string.replace(item,\"\")..    def reverse(self):.        return self.string[::-1]            ..    #def reverse(self):.    #    self.string = self.string[::-1]    i am prone to add this method..    def check(self):.        reversed = self.reverse().        if self.string == reversed:.            print(\"yes\").        else:.            print(\"no\")..det = detector(\"rise to vote, sir!\").det.eliminator().det.check()...when i add the commented lines, the valid \"yes\" becomes \"no\"!",
]

# def plot_value_array(i, predictions_array, true_label):
#     true_label = true_label[i]
#     plt.grid(False)
#     plt.xticks(range(10))
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)

#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')

predictions = export_model.predict(examples)
print(predictions)
# class_names = ['C#', 'Java', 'Javascript', 'Python']

# plot_value_array(1, predictions[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)

# plt.show()
