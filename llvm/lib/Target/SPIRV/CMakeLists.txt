add_llvm_component_group(SPIRV)

add_llvm_target(SPIRVCodeGen
  SPIRVTargetMachine.cpp

  LINK_COMPONENTS
  CodeGen
  Core
  SPIRVInfo
  Support
  Target

  ADD_TO_COMPONENT
  SPIRV
  )

add_subdirectory(TargetInfo)
