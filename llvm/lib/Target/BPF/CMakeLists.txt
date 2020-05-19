set(LLVM_TARGET_DEFINITIONS BPF.td)

tablegen(LLVM BPFGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM BPFGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM BPFGenCallingConv.inc -gen-callingconv)
tablegen(LLVM BPFGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM BPFGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM BPFGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM BPFGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM BPFGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM BPFGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(BPFCommonTableGen)

add_llvm_target(BPFCodeGen
  BPFAbstractMemberAccess.cpp
  BPFAsmPrinter.cpp
  BPFFrameLowering.cpp
  BPFInstrInfo.cpp
  BPFISelDAGToDAG.cpp
  BPFISelLowering.cpp
  BPFMCInstLower.cpp
  BPFPreserveDIType.cpp
  BPFRegisterInfo.cpp
  BPFSelectionDAGInfo.cpp
  BPFSubtarget.cpp
  BPFTargetMachine.cpp
  BPFMIPeephole.cpp
  BPFMIChecking.cpp
  BPFMISimplifyPatchable.cpp
  BTFDebug.cpp
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
