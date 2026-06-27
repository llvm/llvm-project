set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND TypeSystem)
# This dependency is part of a loop (TypeSystem<->SymbolFile).
set_property(DIRECTORY PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES SymbolFile)

add_subdirectory(Clang)

if("flang" IN_LIST LLVM_ENABLE_PROJECTS OR "all" IN_LIST LLVM_ENABLE_PROJECTS)
  add_subdirectory(Fortran)
endif()