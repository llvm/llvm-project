set(LLVM_TARGET_DEFINITIONS VE.td)

tablegen(LLVM VEGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM VEGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM VEGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM VEGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM VEGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM VEGenCallingConv.inc -gen-callingconv)
add_public_tablegen_target(VECommonTableGen)

add_llvm_target(VECodeGen
  VEAsmPrinter.cpp
  VEFrameLowering.cpp
  VEISelDAGToDAG.cpp
  VEISelLowering.cpp
  VEInstrInfo.cpp
  VEMachineFunctionInfo.cpp
  VEMCInstLower.cpp
  VERegisterInfo.cpp
  VESubtarget.cpp
  VETargetMachine.cpp
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
