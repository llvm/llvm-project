add_llvm_component_group(Toy64)

add_llvm_component_library(LLVMToy64CodeGen
  Toy64TargetMachine.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  Toy64
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)