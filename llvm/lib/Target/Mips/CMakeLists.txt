set(LLVM_TARGET_DEFINITIONS Mips.td)

tablegen(LLVM MipsGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MipsGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MipsGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM MipsGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM MipsGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
tablegen(LLVM MipsGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM MipsGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM MipsGenCallingConv.inc -gen-callingconv)
tablegen(LLVM MipsGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MipsGenEDInfo.inc -gen-enhanced-disassembly-info)
tablegen(LLVM MipsGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM MipsGenMCPseudoLowering.inc -gen-pseudo-lowering)
add_public_tablegen_target(MipsCommonTableGen)

add_llvm_target(MipsCodeGen
  Mips16FrameLowering.cpp
  Mips16InstrInfo.cpp
  Mips16RegisterInfo.cpp
  MipsAnalyzeImmediate.cpp
  MipsAsmPrinter.cpp
  MipsCodeEmitter.cpp
  MipsDelaySlotFiller.cpp
  MipsJITInfo.cpp
  MipsInstrInfo.cpp
  MipsISelDAGToDAG.cpp
  MipsISelLowering.cpp
  MipsFrameLowering.cpp
  MipsLongBranch.cpp
  MipsMCInstLower.cpp
  MipsMachineFunction.cpp
  MipsRegisterInfo.cpp
  MipsSEFrameLowering.cpp
  MipsSEInstrInfo.cpp
  MipsSERegisterInfo.cpp
  MipsSubtarget.cpp
  MipsTargetMachine.cpp
  MipsTargetObjectFile.cpp
  MipsSelectionDAGInfo.cpp
  )

add_dependencies(LLVMMipsCodeGen intrinsics_gen)

add_subdirectory(InstPrinter)
add_subdirectory(Disassembler)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(AsmParser)
