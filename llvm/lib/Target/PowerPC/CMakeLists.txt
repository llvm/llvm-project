set(LLVM_TARGET_DEFINITIONS PPC.td)

tablegen(LLVM PPCGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM PPCGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM PPCGenCallingConv.inc -gen-callingconv)
tablegen(LLVM PPCGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM PPCGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM PPCGenFastISel.inc -gen-fast-isel)
tablegen(LLVM PPCGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM PPCGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM PPCGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM PPCGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(PowerPCCommonTableGen)

add_llvm_target(PowerPCCodeGen
  PPCBoolRetToInt.cpp
  PPCAsmPrinter.cpp
  PPCBranchSelector.cpp
  PPCBranchCoalescing.cpp
  PPCCCState.cpp
  PPCCTRLoops.cpp
  PPCHazardRecognizers.cpp
  PPCInstrInfo.cpp
  PPCISelDAGToDAG.cpp
  PPCISelLowering.cpp
  PPCEarlyReturn.cpp
  PPCFastISel.cpp
  PPCFrameLowering.cpp
  PPCLoopPreIncPrep.cpp
  PPCMCInstLower.cpp
  PPCMachineFunctionInfo.cpp
  PPCMIPeephole.cpp
  PPCRegisterInfo.cpp
  PPCQPXLoadSplat.cpp
  PPCSubtarget.cpp
  PPCTargetMachine.cpp
  PPCTargetObjectFile.cpp
  PPCTargetTransformInfo.cpp
  PPCTOCRegDeps.cpp
  PPCTLSDynamicCall.cpp
  PPCVSXCopy.cpp
  PPCReduceCRLogicals.cpp
  PPCVSXFMAMutate.cpp
  PPCVSXSwapRemoval.cpp
  PPCExpandISEL.cpp
  PPCPreEmitPeephole.cpp
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
