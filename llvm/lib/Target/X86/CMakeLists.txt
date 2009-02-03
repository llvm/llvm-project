set(LLVM_TARGET_DEFINITIONS X86.td)

tablegen(X86GenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(X86GenRegisterNames.inc -gen-register-enums)
tablegen(X86GenRegisterInfo.inc -gen-register-desc)
tablegen(X86GenInstrNames.inc -gen-instr-enums)
tablegen(X86GenInstrInfo.inc -gen-instr-desc)
tablegen(X86GenAsmWriter.inc -gen-asm-writer)
tablegen(X86GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
tablegen(X86GenDAGISel.inc -gen-dag-isel)
tablegen(X86GenFastISel.inc -gen-fast-isel)
tablegen(X86GenCallingConv.inc -gen-callingconv)
tablegen(X86GenSubtarget.inc -gen-subtarget)

add_llvm_target(X86CodeGen
  X86CodeEmitter.cpp
  X86ELFWriterInfo.cpp
  X86FloatingPoint.cpp
  X86FloatingPointRegKill.cpp
  X86ISelDAGToDAG.cpp
  X86ISelLowering.cpp
  X86InstrInfo.cpp
  X86JITInfo.cpp
  X86RegisterInfo.cpp
  X86Subtarget.cpp
  X86TargetAsmInfo.cpp
  X86TargetMachine.cpp
  X86FastISel.cpp
  )
