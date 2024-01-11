add_llvm_component_library(LLVMDWARFLinkerParallel
  AcceleratorRecordsSaver.cpp
  DependencyTracker.cpp
  DIEAttributeCloner.cpp
  DWARFEmitterImpl.cpp
  DWARFLinker.cpp
  DWARFLinkerCompileUnit.cpp
  DWARFLinkerTypeUnit.cpp
  DWARFLinkerImpl.cpp
  DWARFLinkerUnit.cpp
  OutputSections.cpp
  SyntheticTypeNameBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DWARFLinkerParallel

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  AsmPrinter
  BinaryFormat
  CodeGen
  DebugInfoDWARF
  DWARFLinkerBase
  MC
  Object
  Support
  TargetParser
  )
