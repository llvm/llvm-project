set(LLVM_TARGET_DEFINITIONS PPC.td)

tablegen(PPCGenInstrNames.inc -gen-instr-enums)
tablegen(PPCGenRegisterNames.inc -gen-register-enums)
tablegen(PPCGenAsmWriter.inc -gen-asm-writer)
tablegen(PPCGenCodeEmitter.inc -gen-emitter)
tablegen(PPCGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(PPCGenRegisterInfo.inc -gen-register-desc)
tablegen(PPCGenInstrInfo.inc -gen-instr-desc)
tablegen(PPCGenDAGISel.inc -gen-dag-isel)
tablegen(PPCGenCallingConv.inc -gen-callingconv)
tablegen(PPCGenSubtarget.inc -gen-subtarget)

add_llvm_target(PowerPCCodeGen
  PPCBranchSelector.cpp
  PPCCodeEmitter.cpp
  PPCHazardRecognizers.cpp
  PPCInstrInfo.cpp
  PPCISelDAGToDAG.cpp
  PPCISelLowering.cpp
  PPCJITInfo.cpp
  PPCMachOWriterInfo.cpp
  PPCPredicates.cpp
  PPCRegisterInfo.cpp
  PPCSubtarget.cpp
  PPCTargetAsmInfo.cpp
  PPCTargetMachine.cpp
  )
