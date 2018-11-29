add_llvm_library(LLVMTextAPI
  MachO/Architecture.cpp
  MachO/ArchitectureSet.cpp
  MachO/InterfaceFile.cpp
  MachO/PackedVersion.cpp
  MachO/Symbol.cpp
  MachO/TextStub.cpp
  MachO/TextStubCommon.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/TextAPI"
)
