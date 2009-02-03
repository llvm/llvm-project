set(LLVM_TARGET_DEFINITIONS IA64.td)

tablegen(IA64GenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(IA64GenRegisterNames.inc -gen-register-enums)
tablegen(IA64GenRegisterInfo.inc -gen-register-desc)
tablegen(IA64GenInstrNames.inc -gen-instr-enums)
tablegen(IA64GenInstrInfo.inc -gen-instr-desc)
tablegen(IA64GenAsmWriter.inc -gen-asm-writer)
tablegen(IA64GenDAGISel.inc -gen-dag-isel)

add_llvm_target(IA64
  IA64AsmPrinter.cpp
  IA64Bundling.cpp
  IA64InstrInfo.cpp
  IA64ISelDAGToDAG.cpp
  IA64ISelLowering.cpp
  IA64RegisterInfo.cpp
  IA64Subtarget.cpp
  IA64TargetAsmInfo.cpp
  IA64TargetMachine.cpp
  )
