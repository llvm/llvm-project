set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND ExpressionParser)

add_subdirectory(Clang)
if(LLDB_ENABLE_SWIFT_SUPPORT)
  add_subdirectory(Swift)
endif()
