# On macOS, system python binaries like /usr/bin/python and $(xcrun -f python3)
# are shims. They do some light validation work and then spawn the "real" python
# binary. Find the "real" python by asking dyld -- sys.executable reports the
# wrong thing more often than not. This is also useful when we're running under
# a Homebrew python3 binary, which also appears to be some kind of shim.
def getDarwinRealPythonExecutable():
    import ctypes
    dyld = ctypes.cdll.LoadLibrary('/usr/lib/system/libdyld.dylib')
    namelen = ctypes.c_ulong(1024)
    name = ctypes.create_string_buffer(b'\000', namelen.value)
    dyld._NSGetExecutablePath(ctypes.byref(name), ctypes.byref(namelen))
    return name.value.decode('utf-8').strip()

print(getDarwinRealPythonExecutable())
