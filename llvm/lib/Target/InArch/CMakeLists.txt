add_llvm_component_group(InArch)

add_llvm_target(InArchCodeGen
  InArchTargetMachine.cpp

  LINK_COMPONENTS
  CodeGen
  Core
  Support
  Target
  InArchInfo

  ADD_TO_COMPONENT
  InArch
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)