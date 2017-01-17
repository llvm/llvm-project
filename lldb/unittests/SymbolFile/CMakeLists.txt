add_subdirectory(DWARF)
if (LLVM_ENABLE_DIA_SDK)
  add_subdirectory(PDB)
endif()
