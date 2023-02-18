add_llvm_component_library(LLVMTextAPI
  Architecture.cpp
  ArchitectureSet.cpp
  InterfaceFile.cpp
  TextStubV5.cpp
  PackedVersion.cpp
  Platform.cpp
  Symbol.cpp
  Target.cpp
  TextStub.cpp
  TextStubCommon.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/TextAPI"

  LINK_COMPONENTS
  Support
  BinaryFormat
  TargetParser
)
