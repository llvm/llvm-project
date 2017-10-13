set(LLVM_TARGET_DEFINITIONS Hexagon.td)

tablegen(LLVM HexagonGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM HexagonGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM HexagonGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM HexagonGenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM HexagonGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM HexagonGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM HexagonGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM HexagonGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM HexagonGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(HexagonCommonTableGen)

add_llvm_target(HexagonCodeGen
  BitTracker.cpp
  HexagonAsmPrinter.cpp
  HexagonBitSimplify.cpp
  HexagonBitTracker.cpp
  HexagonBlockRanges.cpp
  HexagonBranchRelaxation.cpp
  HexagonCFGOptimizer.cpp
  HexagonCommonGEP.cpp
  HexagonConstExtenders.cpp
  HexagonConstPropagation.cpp
  HexagonCopyToCombine.cpp
  HexagonEarlyIfConv.cpp
  HexagonExpandCondsets.cpp
  HexagonFixupHwLoops.cpp
  HexagonFrameLowering.cpp
  HexagonGenExtract.cpp
  HexagonGenInsert.cpp
  HexagonGenMux.cpp
  HexagonGenPredicate.cpp
  HexagonHardwareLoops.cpp
  HexagonHazardRecognizer.cpp
  HexagonInstrInfo.cpp
  HexagonISelDAGToDAG.cpp
  HexagonISelLowering.cpp
  HexagonLoopIdiomRecognition.cpp
  HexagonMachineFunctionInfo.cpp
  HexagonMachineScheduler.cpp
  HexagonMCInstLower.cpp
  HexagonNewValueJump.cpp
  HexagonOptAddrMode.cpp
  HexagonOptimizeSZextends.cpp
  HexagonPeephole.cpp
  HexagonRDFOpt.cpp
  HexagonRegisterInfo.cpp
  HexagonSelectionDAGInfo.cpp
  HexagonSplitConst32AndConst64.cpp
  HexagonSplitDouble.cpp
  HexagonStoreWidening.cpp
  HexagonSubtarget.cpp
  HexagonTargetMachine.cpp
  HexagonTargetObjectFile.cpp
  HexagonTargetTransformInfo.cpp
  HexagonVectorLoopCarriedReuse.cpp
  HexagonVectorPrint.cpp
  HexagonVLIWPacketizer.cpp
  RDFCopy.cpp
  RDFDeadCode.cpp
  RDFGraph.cpp
  RDFLiveness.cpp
  RDFRegisters.cpp
)

add_subdirectory(AsmParser)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(Disassembler)
