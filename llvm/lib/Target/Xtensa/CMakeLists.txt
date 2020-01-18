set(LLVM_TARGET_DEFINITIONS Xtensa.td)

tablegen(LLVM XtensaGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM XtensaGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM XtensaGenCallingConv.inc -gen-callingconv)
tablegen(LLVM XtensaGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM XtensaGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM XtensaGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM XtensaGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM XtensaGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM XtensaGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(XtensaCommonTableGen) 

add_llvm_target(XtensaCodeGen
  XtensaAsmPrinter.cpp
  XtensaConstantPoolValue.cpp
  XtensaFrameLowering.cpp
  XtensaInstrInfo.cpp
  XtensaISelDAGToDAG.cpp
  XtensaISelLowering.cpp
  XtensaMachineFunctionInfo.cpp
  XtensaMCInstLower.cpp
  XtensaRegisterInfo.cpp
  XtensaSizeReductionPass.cpp
  XtensaSubtarget.cpp
  XtensaTargetMachine.cpp
  XtensaTargetObjectFile.cpp
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)

