set(LLVM_TARGET_DEFINITIONS Sparc.td)

tablegen(SparcGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(SparcGenRegisterNames.inc -gen-register-enums)
tablegen(SparcGenRegisterInfo.inc -gen-register-desc)
tablegen(SparcGenInstrNames.inc -gen-instr-enums)
tablegen(SparcGenInstrInfo.inc -gen-instr-desc)
tablegen(SparcGenAsmWriter.inc -gen-asm-writer)
tablegen(SparcGenDAGISel.inc -gen-dag-isel)
tablegen(SparcGenSubtarget.inc -gen-subtarget)
tablegen(SparcGenCallingConv.inc -gen-callingconv)

add_llvm_target(SparcCodeGen
  DelaySlotFiller.cpp
  FPMover.cpp
  SparcInstrInfo.cpp
  SparcISelDAGToDAG.cpp
  SparcISelLowering.cpp
  SparcRegisterInfo.cpp
  SparcSubtarget.cpp
  SparcTargetAsmInfo.cpp
  SparcTargetMachine.cpp
  )
