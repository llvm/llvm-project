add_llvm_component_group(SC32)

set(LLVM_TARGET_DEFINITIONS SC32.td)

tablegen(LLVM SC32GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM SC32GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SC32GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM SC32GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SC32GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SC32GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM SC32GenSDNodeInfo.inc -gen-sd-node-info -sdnode-namespace=SC32ISD)
tablegen(LLVM SC32GenCallingConv.inc -gen-callingconv)

add_public_tablegen_target(SC32CommonTableGen)

add_llvm_target(SC32CodeGen
  SC32TargetMachine.cpp
  SC32PassConfig.cpp
  SC32AsmPrinter.cpp
  SC32Subtarget.cpp
  SC32RegisterInfo.cpp
  SC32FrameLowering.cpp
  SC32InstrInfo.cpp
  SC32DAGToDAGISelLegacy.cpp
  SC32DAGToDAGISel.cpp
  SC32SelectionDAGInfo.cpp
  SC32TargetLowering.cpp

  LINK_COMPONENTS
  SC32Info
  SC32Desc
  Support
  CodeGen
  Target
  TargetParser
  Core
  AsmPrinter
  MC
  SelectionDAG

  ADD_TO_COMPONENT
  SC32
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
