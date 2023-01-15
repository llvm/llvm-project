add_llvm_component_library(LLVMDWARFLinker
  DWARFLinkerCompileUnit.cpp
  DWARFLinkerDeclContext.cpp
  DWARFLinker.cpp
  DWARFStreamer.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DWARFLinker

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  BinaryFormat
  DebugInfoDWARF
  AsmPrinter
  CodeGen
  MC
  Object
  Support
  TargetParser
  )
