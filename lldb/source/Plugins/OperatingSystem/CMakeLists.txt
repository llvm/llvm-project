set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND OperatingSystem)

if (LLDB_ENABLE_PYTHON)
  add_subdirectory(Python)
endif()
