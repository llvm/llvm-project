add_llvm_component_group(MyLoongArch)

add_llvm_target(MyLoongArchCodeGen
  MyLoongArchTargetMachine.cpp

  # LINK_COMPONENTS
  # MyLoongArchDesc
  # MyLoongArchInfo

  ADD_TO_COMPONENT
  MyLoongArch  
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)