set(LLVM_TARGET_DEFINITIONS Hexagon.td)

tablegen(LLVM HexagonGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM HexagonGenCallingConv.inc -gen-callingconv)
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
  HexagonCFGOptimizer.cpp
  HexagonCommonGEP.cpp
  HexagonCopyToCombine.cpp
  HexagonEarlyIfConv.cpp
  HexagonExpandCondsets.cpp
  HexagonExpandPredSpillCode.cpp
  HexagonFixupHwLoops.cpp
  HexagonFrameLowering.cpp
  HexagonGenExtract.cpp
  HexagonGenInsert.cpp
  HexagonGenMux.cpp
  HexagonGenPredicate.cpp
  HexagonHardwareLoops.cpp
  HexagonInstrInfo.cpp
  HexagonISelDAGToDAG.cpp
  HexagonISelLowering.cpp
  HexagonMachineFunctionInfo.cpp
  HexagonMachineScheduler.cpp
  HexagonMCInstLower.cpp
  HexagonNewValueJump.cpp
  HexagonOptimizeSZextends.cpp
  HexagonPeephole.cpp
  HexagonRegisterInfo.cpp
  HexagonSelectionDAGInfo.cpp
  HexagonSplitConst32AndConst64.cpp
  HexagonSplitDouble.cpp
  HexagonStoreWidening.cpp
  HexagonSubtarget.cpp
  HexagonTargetMachine.cpp
  HexagonTargetObjectFile.cpp
  HexagonTargetTransformInfo.cpp
  HexagonVLIWPacketizer.cpp
)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(Disassembler)
