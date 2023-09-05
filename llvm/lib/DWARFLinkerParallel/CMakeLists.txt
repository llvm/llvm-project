add_llvm_component_library(LLVMDWARFLinkerParallel
  DependencyTracker.cpp
  DIEAttributeCloner.cpp
  DWARFEmitterImpl.cpp
  DWARFFile.cpp
  DWARFLinker.cpp
  DWARFLinkerCompileUnit.cpp
  DWARFLinkerImpl.cpp
  DWARFLinkerUnit.cpp
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
