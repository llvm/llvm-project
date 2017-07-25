set(LLVM_TARGET_DEFINITIONS AVR.td)

tablegen(LLVM AVRGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AVRGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AVRGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AVRGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM AVRGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM AVRGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AVRGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AVRGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AVRGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AVRGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AVRGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM AVRGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AVRGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AVRGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(AVRCommonTableGen)

add_llvm_target(AVRCodeGen
  AVRAsmPrinter.cpp
  AVRExpandPseudoInsts.cpp
  AVRFrameLowering.cpp
  AVRInstrInfo.cpp
  AVRISelDAGToDAG.cpp
  AVRISelLowering.cpp
  AVRMCInstLower.cpp
  AVRRelaxMemOperations.cpp
  AVRRegisterInfo.cpp
  AVRSubtarget.cpp
  AVRTargetMachine.cpp
  AVRTargetObjectFile.cpp

  DEPENDS
  intrinsics_gen
)

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
