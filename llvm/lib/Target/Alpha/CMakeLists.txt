set(LLVM_TARGET_DEFINITIONS Alpha.td)

tablegen(AlphaGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(AlphaGenRegisterNames.inc -gen-register-enums)
tablegen(AlphaGenRegisterInfo.inc -gen-register-desc)
tablegen(AlphaGenInstrNames.inc -gen-instr-enums)
tablegen(AlphaGenInstrInfo.inc -gen-instr-desc)
tablegen(AlphaGenCodeEmitter.inc -gen-emitter)
tablegen(AlphaGenAsmWriter.inc -gen-asm-writer)
tablegen(AlphaGenDAGISel.inc -gen-dag-isel)
tablegen(AlphaGenSubtarget.inc -gen-subtarget)

add_llvm_target(AlphaCodeGen
  AlphaBranchSelector.cpp
  AlphaCodeEmitter.cpp
  AlphaInstrInfo.cpp
  AlphaISelDAGToDAG.cpp
  AlphaISelLowering.cpp
  AlphaJITInfo.cpp
  AlphaLLRP.cpp
  AlphaRegisterInfo.cpp
  AlphaSubtarget.cpp
  AlphaTargetAsmInfo.cpp
  AlphaTargetMachine.cpp
  )
