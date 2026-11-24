add_llvm_component_group(Xtensa)

set(LLVM_TARGET_DEFINITIONS Xtensa.td)

llvm_tablegen(XtensaGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(XtensaGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(XtensaGenCallingConv.inc -gen-callingconv)
llvm_tablegen(XtensaGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(XtensaGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(XtensaGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(XtensaGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(XtensaGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(XtensaGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(XtensaGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(XtensaCommonTableGen)

add_llvm_target(XtensaCodeGen
  XtensaAsmPrinter.cpp
  XtensaConstantPoolValue.cpp
  XtensaFrameLowering.cpp
  XtensaInstrInfo.cpp
  XtensaISelDAGToDAG.cpp
  XtensaISelLowering.cpp
  XtensaRegisterInfo.cpp
  XtensaSelectionDAGInfo.cpp
  XtensaSubtarget.cpp
  XtensaTargetMachine.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  XtensaDesc
  XtensaInfo

  ADD_TO_COMPONENT
  Xtensa
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)

