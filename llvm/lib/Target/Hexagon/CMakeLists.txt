set(LLVM_TARGET_DEFINITIONS Hexagon.td)

tablegen(LLVM HexagonGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM HexagonGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM HexagonGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM HexagonGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM HexagonGenCallingConv.inc -gen-callingconv)
tablegen(LLVM HexagonGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM HexagonGenDFAPacketizer.inc -gen-dfa-packetizer)
add_public_tablegen_target(HexagonCommonTableGen)

add_llvm_target(HexagonCodeGen
  HexagonAsmPrinter.cpp
  HexagonCallingConvLower.cpp
  HexagonCFGOptimizer.cpp
  HexagonExpandPredSpillCode.cpp
  HexagonFrameLowering.cpp
  HexagonHardwareLoops.cpp
  HexagonMachineScheduler.cpp
  HexagonMCInstLower.cpp
  HexagonInstrInfo.cpp
  HexagonISelDAGToDAG.cpp
  HexagonISelLowering.cpp
  HexagonPeephole.cpp
  HexagonRegisterInfo.cpp
  HexagonRemoveSZExtArgs.cpp
  HexagonSelectionDAGInfo.cpp
  HexagonSplitTFRCondSets.cpp
  HexagonSubtarget.cpp
  HexagonTargetMachine.cpp
  HexagonTargetObjectFile.cpp
  HexagonVLIWPacketizer.cpp
  HexagonNewValueJump.cpp
)

add_dependencies(LLVMHexagonCodeGen intrinsics_gen)

add_subdirectory(TargetInfo)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)

