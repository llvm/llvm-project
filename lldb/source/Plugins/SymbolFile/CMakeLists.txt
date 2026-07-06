set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND SymbolFile)
set_property(DIRECTORY PROPERTY LLDB_ACCEPTABLE_PLUGIN_DEPENDENCIES ObjectFile)
set_property(DIRECTORY PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES
  Language
  TypeSystem  # part of a loop (TypeSystem<->SymbolFile).
)

add_subdirectory(Breakpad)
add_subdirectory(CTF)
add_subdirectory(DWARF)
add_subdirectory(JSON)
add_subdirectory(NativePDB)
add_subdirectory(PDB)
add_subdirectory(Symtab)
