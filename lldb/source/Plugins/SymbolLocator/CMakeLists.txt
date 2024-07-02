# Order matters here: the first symbol locator prevents further searching.
# For DWARF binaries that are both stripped and split, the Default plugin
# will return the stripped binary when asked for the ObjectFile, which then
# prevents an unstripped binary from being requested from the Debuginfod
# provider.
add_subdirectory(Debuginfod)
add_subdirectory(Default)
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_subdirectory(DebugSymbols)
endif()
