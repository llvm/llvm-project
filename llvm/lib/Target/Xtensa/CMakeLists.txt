add_llvm_component_group(Xtensa)

set(LLVM_TARGET_DEFINITIONS Xtensa.td)

tablegen(LLVM XtensaGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM XtensaGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM XtensaGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM XtensaGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM XtensaGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM XtensaGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM XtensaGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(XtensaCommonTableGen)

add_llvm_target(XtensaCodeGen
  XtensaTargetMachine.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  Core
  MC
  Support
  Target
  XtensaDesc
  XtensaInfo

  ADD_TO_COMPONENT
  Xtensa
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)

