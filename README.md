## Switching from one backend to another

If you have run Keras at least once, you will find the Keras configuration file at:

`$HOME/.keras/keras.json`

If it isn't there, you can create it.

NOTE for Windows Users: Please replace `$HOME` with `%USERPROFILE%.`

The default configuration file looks like this:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```