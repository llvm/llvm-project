add_llvm_component_group(LoongArch)

add_llvm_target(LoongArchCodeGen
  LoongArchTargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  MC
  LoongArchDesc
  LoongArchInfo
  SelectionDAG
  Support
  Target
  GlobalISel

  ADD_TO_COMPONENT
  LoongArch
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
