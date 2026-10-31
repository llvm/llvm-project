add_llvm_component_group(RISCV)

set(LLVM_TARGET_DEFINITIONS RISCV.td)

llvm_tablegen(RISCVGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(RISCVGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(RISCVGenCompressInstEmitter.inc -gen-compress-inst-emitter)
llvm_tablegen(RISCVGenMacroFusion.inc -gen-macro-fusion-pred)
llvm_tablegen(RISCVGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(RISCVGenDisassemblerTables.inc -gen-disassembler
              --specialize-decoders-per-bitwidth)
llvm_tablegen(RISCVGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(RISCVGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(RISCVGenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(RISCVGenRegisterBank.inc -gen-register-bank)
llvm_tablegen(RISCVGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(RISCVGenSearchableTables.inc -gen-searchable-tables)
llvm_tablegen(RISCVGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(RISCVGenExegesis.inc -gen-exegesis)
llvm_tablegen(RISCVGenSDNodeInfo.inc -gen-sd-node-info)

set(LLVM_TARGET_DEFINITIONS RISCVGISel.td)
llvm_tablegen(RISCVGenGlobalISel.inc -gen-global-isel)
llvm_tablegen(RISCVGenO0PreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="RISCVO0PreLegalizerCombiner")
llvm_tablegen(RISCVGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="RISCVPreLegalizerCombiner")
llvm_tablegen(RISCVGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="RISCVPostLegalizerCombiner")

set(LLVM_TARGET_DEFINITIONS RISCVOptions.td)
llvm_tablegen(RISCVOptions.inc -gen-opt-parser-defs)

set(LLVM_TARGET_DEFINITIONS MCTargetDesc/RISCVMCOptions.td)
llvm_tablegen(RISCVMCOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(RISCVCommonTableGen)

add_llvm_target(RISCVCodeGen
  RISCVAsmPrinter.cpp
  RISCVCallingConv.cpp
  RISCVCodeGenPassBuilder.cpp
  RISCVCodeGenPrepare.cpp
  RISCVConstantPoolValue.cpp
  RISCVDeadRegisterDefinitions.cpp
  RISCVExpandPseudoAtomics.cpp
  RISCVExpandPseudoBase.cpp
  RISCVExpandPseudoPostRA.cpp
  RISCVExpandPseudoPreEmit.cpp
  RISCVExpandPseudoPreRA.cpp
  RISCVFoldMemOffset.cpp
  RISCVFrameLowering.cpp
  RISCVGatherScatterLowering.cpp
  RISCVIndirectBranchTracking.cpp
  RISCVInsertReadWriteCSR.cpp
  RISCVInsertVSETVLI.cpp
  RISCVInsertWriteVXRM.cpp
  RISCVInstrInfo.cpp
  RISCVInterleavedAccess.cpp
  RISCVISelDAGToDAG.cpp
  RISCVISelLowering.cpp
  RISCVLandingPadSetup.cpp
  RISCVLateBranchOpt.cpp
  RISCVLoadStoreOptimizer.cpp
  RISCVMachineFunctionInfo.cpp
  RISCVMakeCompressible.cpp
  RISCVMachineScheduler.cpp
  RISCVMergeBaseOffset.cpp
  RISCVMoveMerger.cpp
  RISCVOptWInstrs.cpp
  RISCVPromoteConstant.cpp
  RISCVPushPopOptimizer.cpp
  RISCVQCRelaxMarking.cpp
  RISCVRedundantCopyElimination.cpp
  RISCVRegisterInfo.cpp
  RISCVSelectionDAGInfo.cpp
  RISCVSubtarget.cpp
  RISCVTargetMachine.cpp
  RISCVTargetObjectFile.cpp
  RISCVTargetTransformInfo.cpp
  RISCVVectorMaskDAGMutation.cpp
  RISCVVectorPeephole.cpp
  RISCVVLOptimizer.cpp
  RISCVVMV0Elimination.cpp
  RISCVVSETVLIInfoAnalysis.cpp
  RISCVZacasABIFix.cpp
  RISCVZilsdOptimizer.cpp
  GISel/RISCVCallLowering.cpp
  GISel/RISCVInlineAsmLowering.cpp
  GISel/RISCVInstructionSelector.cpp
  GISel/RISCVLegalizerInfo.cpp
  GISel/RISCVPostLegalizerCombiner.cpp
  GISel/RISCVO0PreLegalizerCombiner.cpp
  GISel/RISCVPreLegalizerCombiner.cpp
  GISel/RISCVRegisterBankInfo.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  IPO
  MC
  Option
  Passes
  RISCVDesc
  RISCVInfo
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize

  ADD_TO_COMPONENT
  RISCV
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(MCA)
add_subdirectory(TargetInfo)
