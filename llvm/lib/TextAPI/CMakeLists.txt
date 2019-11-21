add_llvm_component_library(LLVMTextAPI
  ELF/ELFStub.cpp
  ELF/TBEHandler.cpp
  MachO/Architecture.cpp
  MachO/ArchitectureSet.cpp
  MachO/InterfaceFile.cpp
  MachO/PackedVersion.cpp
  MachO/Platform.cpp
  MachO/Symbol.cpp
  MachO/Target.cpp
  MachO/TextStub.cpp
  MachO/TextStubCommon.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/TextAPI"
)
