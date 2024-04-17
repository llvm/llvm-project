def no_libsanitizers(testbase):
    testbase.runCmd("image list libsystem_sanitizers.dylib", check=False)
    return not "libsystem_sanitizers.dylib" in testbase.res.GetOutput()
