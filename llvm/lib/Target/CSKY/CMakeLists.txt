add_llvm_component_group(CSKY)

add_llvm_target(CSKYCodeGen
  CSKYTargetMachine.cpp

  LINK_COMPONENTS
  Core
  CodeGen
  CSKYInfo
  Support
  Target

  ADD_TO_COMPONENT
  CSKY
  )

add_subdirectory(TargetInfo)
