add_llvm_component_group(AMDGPU)

set(LLVM_TARGET_DEFINITIONS AMDGPU.td)

llvm_tablegen(AMDGPUGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(AMDGPUGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(AMDGPUGenCallingConv.inc -gen-callingconv)
llvm_tablegen(AMDGPUGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(AMDGPUGenDisassemblerTables.inc -gen-disassembler
              --specialize-decoders-per-bitwidth
              -ignore-non-decodable-operands
              -ignore-fully-defined-operands)
llvm_tablegen(AMDGPUGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(AMDGPUGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(AMDGPUGenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(AMDGPUGenRegisterBank.inc -gen-register-bank)
llvm_tablegen(AMDGPUGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(AMDGPUGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(AMDGPUGenSearchableTables.inc -gen-searchable-tables)
llvm_tablegen(AMDGPUGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS AMDGPUGISel.td)
llvm_tablegen(AMDGPUGenGlobalISel.inc -gen-global-isel -gisel-extended-llt)
llvm_tablegen(AMDGPUGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPUPreLegalizerCombiner" -gisel-extended-llt)
llvm_tablegen(AMDGPUGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPUPostLegalizerCombiner" -gisel-extended-llt)
llvm_tablegen(AMDGPUGenRegBankGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPURegBankCombiner" -gisel-extended-llt)

set(LLVM_TARGET_DEFINITIONS R600.td)
llvm_tablegen(R600GenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(R600GenCallingConv.inc -gen-callingconv)
llvm_tablegen(R600GenDAGISel.inc -gen-dag-isel)
llvm_tablegen(R600GenDFAPacketizer.inc -gen-dfa-packetizer)
llvm_tablegen(R600GenInstrInfo.inc -gen-instr-info)
llvm_tablegen(R600GenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(R600GenRegisterInfo.inc -gen-register-info)
llvm_tablegen(R600GenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(AMDGPUCommonTableGen)

add_llvm_target(AMDGPUCodeGen
  AMDGPUAliasAnalysis.cpp
  AMDGPUAlwaysInlinePass.cpp
  AMDGPUAnnotateUniformValues.cpp
  AMDGPUArgumentUsageInfo.cpp
  AMDGPUAsanInstrumentation.cpp
  AMDGPUAsmPrinter.cpp
  AMDGPUAtomicOptimizer.cpp
  AMDGPUAttributor.cpp
  AMDGPUBarrierLatency.cpp
  AMDGPUCallLowering.cpp
  AMDGPUCodeGenPrepare.cpp
  AMDGPUCombinerHelper.cpp
  AMDGPUCtorDtorLowering.cpp
  AMDGPUExportClustering.cpp
  AMDGPUExportKernelRuntimeHandles.cpp
  AMDGPUFrameLowering.cpp
  AMDGPUGlobalISelDivergenceLowering.cpp
  AMDGPUGlobalISelUtils.cpp
  AMDGPUHazardLatency.cpp
  AMDGPUHSAMetadataStreamer.cpp
  AMDGPUHWEvents.cpp
  AMDGPUInsertDelayAlu.cpp
  AMDGPUInstCombineIntrinsic.cpp
  AMDGPUUniformIntrinsicCombine.cpp
  AMDGPUInstrInfo.cpp
  AMDGPUInstructionSelector.cpp
  AMDGPUISelDAGToDAG.cpp
  AMDGPUISelLowering.cpp
  AMDGPULateCodeGenPrepare.cpp
  AMDGPULegalizerInfo.cpp
  AMDGPULibCalls.cpp
  AMDGPUImageIntrinsicOptimizer.cpp
  AMDGPULibFunc.cpp
  AMDGPULowerBufferFatPointers.cpp
  AMDGPULowerIntrinsics.cpp
  AMDGPULowerKernelArguments.cpp
  AMDGPULowerKernelAttributes.cpp
  AMDGPULowerModuleLDSPass.cpp
  AMDGPUPrepareAGPRAlloc.cpp
  AMDGPULowerExecSync.cpp
  AMDGPUSwLowerLDS.cpp
  AMDGPUMachineFunctionInfo.cpp
  AMDGPUMachineModuleInfo.cpp
  AMDGPUMacroFusion.cpp
  AMDGPUMCInstLower.cpp
  AMDGPUMemoryUtils.cpp
  AMDGPUCoExecSchedStrategy.cpp
  AMDGPUIGroupLP.cpp
  AMDGPULowerVGPREncoding.cpp
  AMDGPUMCResourceInfo.cpp
  AMDGPUMarkLastScratchLoad.cpp
  AMDGPUMIRFormatter.cpp
  AMDGPUNextUseAnalysis.cpp
  AMDGPUPerfHintAnalysis.cpp
  AMDGPUPostLegalizerCombiner.cpp
  AMDGPUPreLegalizerCombiner.cpp
  AMDGPUPreloadKernArgProlog.cpp
  AMDGPUPreloadKernelArguments.cpp
  AMDGPUPrintfRuntimeBinding.cpp
  AMDGPUPromoteAlloca.cpp
  AMDGPUPromoteKernelArguments.cpp
  AMDGPURegBankCombiner.cpp
  AMDGPURegBankLegalize.cpp
  AMDGPURegBankLegalizeHelper.cpp
  AMDGPURegBankLegalizeRules.cpp
  AMDGPURegBankSelect.cpp
  AMDGPURegisterBankInfo.cpp
  AMDGPURemoveIncompatibleFunctions.cpp
  AMDGPUReserveWWMRegs.cpp
  AMDGPUResourceUsageAnalysis.cpp
  AMDGPURewriteAGPRCopyMFMA.cpp
  AMDGPURewriteOutArguments.cpp
  AMDGPURewriteUndefForPHI.cpp
  AMDGPUSelectionDAGInfo.cpp
  AMDGPUSetWavePriority.cpp
  AMDGPUSplitModule.cpp
  AMDGPUSubtarget.cpp
  AMDGPUTargetMachine.cpp
  AMDGPUTargetObjectFile.cpp
  AMDGPUTargetTransformInfo.cpp
  AMDGPUWaitcntUtils.cpp
  AMDGPUWaitSGPRHazards.cpp
  AMDGPUUnifyDivergentExitNodes.cpp
  R600MachineCFGStructurizer.cpp
  GCNCreateVOPD.cpp
  GCNDPPCombine.cpp
  GCNHazardRecognizer.cpp
  GCNILPSched.cpp
  GCNIterativeScheduler.cpp
  GCNMinRegStrategy.cpp
  GCNNSAReassign.cpp
  GCNPreRAAntiHints.cpp
  GCNPreRAOptimizations.cpp
  GCNPreRALongBranchReg.cpp
  GCNRegPressure.cpp
  GCNRewritePartialRegUses.cpp
  GCNSchedStrategy.cpp
  GCNSubtarget.cpp
  GCNVOPDUtils.cpp
  R600AsmPrinter.cpp
  R600ClauseMergePass.cpp
  R600ControlFlowFinalizer.cpp
  R600EmitClauseMarkers.cpp
  R600ExpandSpecialInstrs.cpp
  R600FrameLowering.cpp
  R600InstrInfo.cpp
  R600ISelDAGToDAG.cpp
  R600ISelLowering.cpp
  R600MachineFunctionInfo.cpp
  R600MachineScheduler.cpp
  R600MCInstLower.cpp
  R600OpenCLImageTypeLoweringPass.cpp
  R600OptimizeVectorRegisters.cpp
  R600Packetizer.cpp
  R600RegisterInfo.cpp
  R600Subtarget.cpp
  R600TargetMachine.cpp
  R600TargetTransformInfo.cpp
  SIAnnotateControlFlow.cpp
  SIFixSGPRCopies.cpp
  SIFixVGPRCopies.cpp
  SIFoldOperands.cpp
  SIFormMemoryClauses.cpp
  SIFrameLowering.cpp
  SIInsertHardClauses.cpp
  SIInsertWaitcnts.cpp
  SIInstrInfo.cpp
  SIISelLowering.cpp
  SILateBranchLowering.cpp
  SILoadStoreOptimizer.cpp
  SILowerControlFlow.cpp
  SILowerI1Copies.cpp
  SILowerWWMCopies.cpp
  SILowerSGPRSpills.cpp
  SIMachineFunctionInfo.cpp
  SISpillUtils.cpp
  SIMachineScheduler.cpp
  SIMemoryLegalizer.cpp
  SIModeRegister.cpp
  SIModeRegisterDefaults.cpp
  SIOptimizeExecMasking.cpp
  SIOptimizeExecMaskingPreRA.cpp
  SIOptimizeVGPRLiveRange.cpp
  SIPeepholeSDWA.cpp
  SIPostRABundler.cpp
  SIPreAllocateWWMRegs.cpp
  SIPreEmitPeephole.cpp
  SIProgramInfo.cpp
  SIRegisterInfo.cpp
  SIShrinkInstructions.cpp
  SIWholeQuadMode.cpp
  SIPostRA16BitMovFolding.cpp

  LINK_COMPONENTS
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  Analysis
  AsmPrinter
  BinaryFormat
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  HipStdPar
  IPO
  IRPrinter
  Instrumentation
  MC
  MIRParser
  ObjCARC
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize

  ADD_TO_COMPONENT
  AMDGPU
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCA)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
