add_llvm_component_library(LLVMDWARFLinker
  DWARFLinkerCompileUnit.cpp
  DWARFLinkerDeclContext.cpp
  DWARFLinker.cpp
  DWARFStreamer.cpp

  DEPENDS
  intrinsics_gen
  )
