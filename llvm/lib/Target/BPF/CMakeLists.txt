set(LLVM_TARGET_DEFINITIONS BPF.td)

tablegen(LLVM BPFGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM BPFGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM BPFGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM BPFGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM BPFGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM BPFGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM BPFGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM BPFGenCallingConv.inc -gen-callingconv)
tablegen(LLVM BPFGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(BPFCommonTableGen)

add_llvm_target(BPFCodeGen
  BPFAsmPrinter.cpp
  BPFFrameLowering.cpp
  BPFInstrInfo.cpp
  BPFISelDAGToDAG.cpp
  BPFISelLowering.cpp
  BPFMCInstLower.cpp
  BPFRegisterInfo.cpp
  BPFSubtarget.cpp
  BPFTargetMachine.cpp
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
