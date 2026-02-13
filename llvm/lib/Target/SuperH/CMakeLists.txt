add_llvm_component_group(SuperH)

set(LLVM_TARGET_DEFINITIONS SuperH.td)

# add_public_tablegen_target(SuperHCommonTableGen)

add_llvm_target(SuperHCodeGen
  SuperHTargetMachine.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser

  ADD_TO_COMPONENT
  SuperH
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)