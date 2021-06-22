add_llvm_component_group(AMDGPU)

set(LLVM_TARGET_DEFINITIONS AMDGPU.td)

tablegen(LLVM AMDGPUGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AMDGPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AMDGPUGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AMDGPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AMDGPUGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM AMDGPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AMDGPUGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM AMDGPUGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM AMDGPUGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM AMDGPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AMDGPUGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM AMDGPUGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS AMDGPUGISel.td)
tablegen(LLVM AMDGPUGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM AMDGPUGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPUPreLegalizerCombinerHelper")
tablegen(LLVM AMDGPUGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPUPostLegalizerCombinerHelper")
tablegen(LLVM AMDGPUGenRegBankGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPURegBankCombinerHelper")

set(LLVM_TARGET_DEFINITIONS R600.td)
tablegen(LLVM R600GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM R600GenCallingConv.inc -gen-callingconv)
tablegen(LLVM R600GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM R600GenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM R600GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM R600GenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM R600GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM R600GenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(AMDGPUCommonTableGen)

set(LLVM_TARGET_DEFINITIONS InstCombineTables.td)
tablegen(LLVM InstCombineTables.inc -gen-searchable-tables)
add_public_tablegen_target(InstCombineTableGen)

add_llvm_target(AMDGPUCodeGen
  AMDGPUAliasAnalysis.cpp
  AMDGPUAlwaysInlinePass.cpp
  AMDGPUAnnotateKernelFeatures.cpp
  AMDGPUAnnotateUniformValues.cpp
  AMDGPUArgumentUsageInfo.cpp
  AMDGPUAsmPrinter.cpp
  AMDGPUAtomicOptimizer.cpp
  AMDGPUCallLowering.cpp
  AMDGPUCodeGenPrepare.cpp
  AMDGPUExportClustering.cpp
  AMDGPUFixFunctionBitcasts.cpp
  AMDGPUFrameLowering.cpp
  AMDGPUHSAMetadataStreamer.cpp
  AMDGPUInstCombineIntrinsic.cpp
  AMDGPUInstrInfo.cpp
  AMDGPUInstructionSelector.cpp
  AMDGPUISelDAGToDAG.cpp
  AMDGPUISelLowering.cpp
  AMDGPUGlobalISelUtils.cpp
  AMDGPULateCodeGenPrepare.cpp
  AMDGPULegalizerInfo.cpp
  AMDGPULibCalls.cpp
  AMDGPULibFunc.cpp
  AMDGPULowerIntrinsics.cpp
  AMDGPULowerKernelArguments.cpp
  AMDGPULowerKernelAttributes.cpp
  AMDGPULowerModuleLDSPass.cpp
  AMDGPUMachineCFGStructurizer.cpp
  AMDGPUMachineFunction.cpp
  AMDGPUMachineModuleInfo.cpp
  AMDGPUMacroFusion.cpp
  AMDGPUMCInstLower.cpp
  AMDGPUMIRFormatter.cpp
  AMDGPUOpenCLEnqueuedBlockLowering.cpp
  AMDGPUPostLegalizerCombiner.cpp
  AMDGPUPreLegalizerCombiner.cpp
  AMDGPUPromoteAlloca.cpp
  AMDGPUPropagateAttributes.cpp
  AMDGPURegBankCombiner.cpp
  AMDGPURegisterBankInfo.cpp
  AMDGPUReplaceLDSUseWithPointer.cpp
  AMDGPURewriteOutArguments.cpp
  AMDGPUSubtarget.cpp
  AMDGPUTargetMachine.cpp
  AMDGPUTargetObjectFile.cpp
  AMDGPUTargetTransformInfo.cpp
  AMDGPUUnifyDivergentExitNodes.cpp
  AMDGPUUnifyMetadata.cpp
  AMDGPUPerfHintAnalysis.cpp
  AMDILCFGStructurizer.cpp
  AMDGPUPrintfRuntimeBinding.cpp
  GCNHazardRecognizer.cpp
  GCNIterativeScheduler.cpp
  GCNMinRegStrategy.cpp
  GCNRegPressure.cpp
  GCNSchedStrategy.cpp
  R600AsmPrinter.cpp
  R600ClauseMergePass.cpp
  R600ControlFlowFinalizer.cpp
  R600EmitClauseMarkers.cpp
  R600ExpandSpecialInstrs.cpp
  R600FrameLowering.cpp
  R600InstrInfo.cpp
  R600ISelLowering.cpp
  R600MachineFunctionInfo.cpp
  R600MachineScheduler.cpp
  R600OpenCLImageTypeLoweringPass.cpp
  R600OptimizeVectorRegisters.cpp
  R600Packetizer.cpp
  R600RegisterInfo.cpp
  SIAnnotateControlFlow.cpp
  SIFixSGPRCopies.cpp
  SIFixVGPRCopies.cpp
  SIPreAllocateWWMRegs.cpp
  SIFoldOperands.cpp
  SIFormMemoryClauses.cpp
  SIFrameLowering.cpp
  SIInsertHardClauses.cpp
  SILateBranchLowering.cpp
  SIInsertWaitcnts.cpp
  SIInstrInfo.cpp
  SIISelLowering.cpp
  SILoadStoreOptimizer.cpp
  SILowerControlFlow.cpp
  SILowerI1Copies.cpp
  SILowerSGPRSpills.cpp
  SIMachineFunctionInfo.cpp
  SIMachineScheduler.cpp
  SIMemoryLegalizer.cpp
  SIOptimizeExecMasking.cpp
  SIOptimizeExecMaskingPreRA.cpp
  SIOptimizeVGPRLiveRange.cpp
  SIPeepholeSDWA.cpp
  SIPostRABundler.cpp
  SIPreEmitPeephole.cpp
  SIProgramInfo.cpp
  SIRegisterInfo.cpp
  SIShrinkInstructions.cpp
  SIWholeQuadMode.cpp
  GCNILPSched.cpp
  GCNNSAReassign.cpp
  GCNDPPCombine.cpp
  SIModeRegister.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  IPO
  MC
  Passes
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  Scalar
  SelectionDAG
  Support
  Target
  TransformUtils
  Vectorize
  GlobalISel
  BinaryFormat
  MIRParser

  ADD_TO_COMPONENT
  AMDGPU
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
