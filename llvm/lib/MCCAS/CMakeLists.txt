add_llvm_component_library(LLVMMCCAS
  MCCASDebugV1.cpp
  MCCASObjectV1.cpp
  MCCASFormatSchemaBase.cpp
  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MCCAS

  LINK_COMPONENTS
  Support
  BinaryFormat
  CAS
  DebugInfoCodeView
  DebugInfoDWARF
  MC
  TargetParser
  )
