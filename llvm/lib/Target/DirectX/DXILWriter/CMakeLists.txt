add_llvm_component_library(LLVMDXILBitWriter
  DXILBitcodeWriter.cpp
  DXILValueEnumerator.cpp
  DXILWriterPass.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Bitwriter
  Core
  MC
  Object
  Support
  )
