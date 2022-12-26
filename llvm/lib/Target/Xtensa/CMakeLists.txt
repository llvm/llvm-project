add_llvm_component_group(Xtensa)

add_llvm_target(XtensaCodeGen
  XtensaTargetMachine.cpp

  LINK_COMPONENTS
  CodeGen
  Core
  Support
  Target
  XtensaInfo

  ADD_TO_COMPONENT
  Xtensa
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
