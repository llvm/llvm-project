set(LLVM_TARGET_DEFINITIONS Lanai.td)

tablegen(LLVM LanaiGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM LanaiGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM LanaiGenCallingConv.inc -gen-callingconv)
tablegen(LLVM LanaiGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM LanaiGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM LanaiGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM LanaiGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM LanaiGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM LanaiGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(LanaiCommonTableGen)

add_llvm_target(LanaiCodeGen
  LanaiAsmPrinter.cpp
  LanaiDelaySlotFiller.cpp
  LanaiFrameLowering.cpp
  LanaiInstrInfo.cpp
  LanaiISelDAGToDAG.cpp
  LanaiISelLowering.cpp
  LanaiMachineFunctionInfo.cpp
  LanaiMCInstLower.cpp
  LanaiMemAluCombiner.cpp
  LanaiRegisterInfo.cpp
  LanaiSelectionDAGInfo.cpp
  LanaiSubtarget.cpp
  LanaiTargetMachine.cpp
  LanaiTargetObjectFile.cpp
)

add_subdirectory(AsmParser)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(InstPrinter)
add_subdirectory(Disassembler)
