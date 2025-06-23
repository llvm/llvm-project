set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND OperatingSystem)

if (LLDB_ENABLE_PYTHON)
  add_subdirectory(Python)
  if (LLDB_ENABLE_SWIFT_SUPPORT)
    add_subdirectory(SwiftTasks)
  endif()
endif()
