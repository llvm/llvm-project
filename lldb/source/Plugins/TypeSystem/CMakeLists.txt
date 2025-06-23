set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND TypeSystem)
# This dependency is part of a loop (TypeSystem<->SymbolFile).
set_property(DIRECTORY PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES SymbolFile)

add_subdirectory(Clang)

if (LLDB_ENABLE_SWIFT_SUPPORT)
  add_subdirectory(Swift)
endif()
