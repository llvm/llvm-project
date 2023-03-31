add_llvm_component_library(LLVMDWARFLinkerParallel
  DWARFEmitterImpl.cpp
  DWARFLinker.cpp
  DWARFLinkerImpl.cpp
  OutputSections.cpp
  StringPool.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DWARFLinkerParallel

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  AsmPrinter
  BinaryFormat
  CodeGen
  DebugInfoDWARF
  MC
  Object
  Support
  TargetParser
  )
