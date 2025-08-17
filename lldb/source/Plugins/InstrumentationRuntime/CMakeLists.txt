set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND InstrumentationRuntime)

add_subdirectory(ASan)
add_subdirectory(ASanLibsanitizers)
add_subdirectory(MainThreadChecker)
add_subdirectory(TSan)
add_subdirectory(UBSan)
add_subdirectory(Utility)
