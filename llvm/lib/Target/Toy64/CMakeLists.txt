set(LLVM_TARGET_DEFINITIONS Toy64.td)

# Generate the C++ register descriptions from Toy64RegisterInfo.td.
tablegen(LLVM Toy64GenRegisterInfo.inc -gen-register-info)

# Group all common generated TableGen files behind one build target.
add_public_tablegen_target(Toy64CommonTableGen)

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