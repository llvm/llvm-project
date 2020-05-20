add_llvm_component_library(LLVMJITLink
  JITLink.cpp
  JITLinkGeneric.cpp
  JITLinkMemoryManager.cpp
  EHFrameSupport.cpp
  MachO.cpp
  MachO_arm64.cpp
  MachO_x86_64.cpp
  MachOLinkGraphBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/JITLink

  DEPENDS
  intrinsics_gen
  )

target_link_libraries(LLVMJITLink
  PRIVATE
  LLVMObject
)
