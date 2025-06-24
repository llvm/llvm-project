set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND Language)
set_property(DIRECTORY PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES
  LanguageRuntime
  TypeSystem
)

add_subdirectory(ClangCommon)
add_subdirectory(CPlusPlus)
add_subdirectory(ObjC)
add_subdirectory(ObjCPlusPlus)
