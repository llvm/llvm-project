set(LLVM_TARGET_DEFINITIONS Sparc.td)

tablegen(LLVM SparcGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SparcGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SparcGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM SparcGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM SparcGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SparcGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM SparcGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM SparcGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM SparcGenCallingConv.inc -gen-callingconv)
add_public_tablegen_target(SparcCommonTableGen)

add_llvm_target(SparcCodeGen
  DelaySlotFiller.cpp
  LeonPasses.cpp
  SparcAsmPrinter.cpp
  SparcInstrInfo.cpp
  SparcISelDAGToDAG.cpp
  SparcISelLowering.cpp
  SparcFrameLowering.cpp
  SparcMachineFunctionInfo.cpp
  SparcRegisterInfo.cpp
  SparcSubtarget.cpp
  SparcTargetMachine.cpp
  SparcMCInstLower.cpp
  SparcTargetObjectFile.cpp
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(InstPrinter)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
