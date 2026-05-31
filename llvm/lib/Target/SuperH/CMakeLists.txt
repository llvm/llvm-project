add_llvm_component_group(SuperH)

set(LLVM_TARGET_DEFINITIONS SuperH.td)

tablegen(LLVM SuperHGenRegisterInfo.inc     -gen-register-info)
#tablegen(LLVM SuperHGenInstrInfo.inc        -gen-instr-info)
tablegen(LLVM SuperHGenSubtargetInfo.inc    -gen-subtarget)

add_public_tablegen_target(SuperHCommonTableGen)

add_llvm_target(SuperHCodeGen
  SuperHTargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  SuperHDesc
  SuperHInfo
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  SuperH
  )

add_subdirectory(AsmParser)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)