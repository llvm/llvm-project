set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND Language)
set_property(DIRECTORY PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES
  LanguageRuntime
  TypeSystem
)

add_subdirectory(CPlusPlus)
add_subdirectory(ObjC)
add_subdirectory(ObjCPlusPlus)
if("flang" IN_LIST LLVM_ENABLE_PROJECTS OR "all" IN_LIST LLVM_ENABLE_PROJECTS)
  add_subdirectory(Fortran)
endif()