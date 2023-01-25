add_llvm_component_library(LLVMDWARFLinkerParallel
  DWARFLinker.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DWARFLinkerParallel

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
)
