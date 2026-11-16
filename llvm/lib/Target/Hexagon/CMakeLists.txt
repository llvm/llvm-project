add_llvm_component_group(Hexagon)

set(LLVM_TARGET_DEFINITIONS Hexagon.td)

llvm_tablegen(HexagonGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(HexagonGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(HexagonGenCallingConv.inc -gen-callingconv)
llvm_tablegen(HexagonGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(HexagonGenDFAPacketizer.inc -gen-dfa-packetizer)
llvm_tablegen(HexagonGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(HexagonGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(HexagonGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(HexagonGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(HexagonGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(HexagonGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(HexagonCommonTableGen)

add_llvm_target(HexagonCodeGen
  BitTracker.cpp
  HexagonAlignGlobalArrays.cpp
  HexagonAsmPrinter.cpp
  HexagonBitSimplify.cpp
  HexagonBitTracker.cpp
  HexagonBlockRanges.cpp
  HexagonBranchRelaxation.cpp
  HexagonCFGOptimizer.cpp
  HexagonCommonGEP.cpp
  HexagonConstExtenders.cpp
  HexagonConstPropagation.cpp
  HexagonCopyHoisting.cpp
  HexagonCopyToCombine.cpp
  HexagonEarlyIfConv.cpp
  HexagonExpandCondsets.cpp
  HexagonFixupHwLoops.cpp
  HexagonFrameLowering.cpp
  HexagonGenExtract.cpp
  HexagonGenInsert.cpp
  HexagonGenMemAbsolute.cpp
  HexagonGenMux.cpp
  HexagonGenPredicate.cpp
  HexagonGlobalRegion.cpp
  HexagonGlobalScheduler.cpp
  HexagonLiveVariables.cpp
  HexagonHardwareLoops.cpp
  HexagonHVXSaveRemark.cpp
  HexagonHazardRecognizer.cpp
  HexagonInstrInfo.cpp
  HexagonISelDAGToDAG.cpp
  HexagonISelDAGToDAGHVX.cpp
  HexagonISelLowering.cpp
  HexagonISelLoweringHVX.cpp
  HexagonLoopAlign.cpp
  HexagonLoopIdiomRecognition.cpp
  HexagonMachineFunctionInfo.cpp
  HexagonMachineScheduler.cpp
  HexagonMask.cpp
  HexagonMCInstLower.cpp
  HexagonNewValueJump.cpp
  HexagonOptAddrMode.cpp
  HexagonOptimizeSZextends.cpp
  HexagonPeephole.cpp
  HexagonPostRAHandleQFP.cpp
  HexagonQFPOptimizer.cpp
  HexagonRDFOpt.cpp
  HexagonRegisterInfo.cpp
  HexagonXQFloatGenerator.cpp
  HexagonSelectionDAGInfo.cpp
  HexagonSplitConst32AndConst64.cpp
  HexagonSplitDouble.cpp
  HexagonLoadStoreWidening.cpp
  HexagonSubtarget.cpp
  HexagonTargetMachine.cpp
  HexagonTargetObjectFile.cpp
  HexagonTargetTransformInfo.cpp
  HexagonTfrCleanup.cpp
  HexagonVectorCombine.cpp
  HexagonVectorLoopCarriedReuse.cpp
  HexagonVectorPrint.cpp
  HexagonVExtract.cpp
  HexagonVLIWPacketizer.cpp
  RDFCopy.cpp
  HexagonAggressiveRDFCopy.cpp
  RDFDeadCode.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  HexagonAsmParser
  HexagonDesc
  HexagonInfo
  IPO
  MC
  MIRParser
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Hexagon
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
