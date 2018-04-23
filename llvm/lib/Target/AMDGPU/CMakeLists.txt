set(LLVM_TARGET_DEFINITIONS AMDGPU.td)

tablegen(LLVM AMDGPUGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AMDGPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AMDGPUGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AMDGPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AMDGPUGenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM AMDGPUGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM AMDGPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AMDGPUGenIntrinsics.inc -gen-tgt-intrinsic)
tablegen(LLVM AMDGPUGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM AMDGPUGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM AMDGPUGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM AMDGPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AMDGPUGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM AMDGPUGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(AMDGPUCommonTableGen)

add_llvm_target(AMDGPUCodeGen
  AMDGPUAliasAnalysis.cpp
  AMDGPUAlwaysInlinePass.cpp
  AMDGPUAnnotateKernelFeatures.cpp
  AMDGPUAnnotateUniformValues.cpp
  AMDGPUArgumentUsageInfo.cpp
  AMDGPUAsmPrinter.cpp
  AMDGPUCallLowering.cpp
  AMDGPUCodeGenPrepare.cpp
  AMDGPUFrameLowering.cpp
  AMDGPUInstrInfo.cpp
  AMDGPUInstructionSelector.cpp
  AMDGPUIntrinsicInfo.cpp
  AMDGPUISelDAGToDAG.cpp
  AMDGPUISelLowering.cpp
  AMDGPULegalizerInfo.cpp
  AMDGPULibCalls.cpp
  AMDGPULibFunc.cpp
  AMDGPULowerIntrinsics.cpp
  AMDGPUMachineCFGStructurizer.cpp
  AMDGPUMachineFunction.cpp
  AMDGPUMachineModuleInfo.cpp
  AMDGPUMacroFusion.cpp
  AMDGPUMCInstLower.cpp
  AMDGPUOpenCLEnqueuedBlockLowering.cpp
  AMDGPUOpenCLImageTypeLoweringPass.cpp
  AMDGPUPromoteAlloca.cpp
  AMDGPURegAsmNames.inc.cpp
  AMDGPURegisterBankInfo.cpp
  AMDGPURegisterInfo.cpp
  AMDGPURewriteOutArguments.cpp
  AMDGPUSubtarget.cpp
  AMDGPUTargetMachine.cpp
  AMDGPUTargetObjectFile.cpp
  AMDGPUTargetTransformInfo.cpp
  AMDGPUUnifyDivergentExitNodes.cpp
  AMDGPUUnifyMetadata.cpp
  AMDGPUInline.cpp
  AMDILCFGStructurizer.cpp
  GCNHazardRecognizer.cpp
  GCNIterativeScheduler.cpp
  GCNMinRegStrategy.cpp
  GCNRegPressure.cpp
  GCNSchedStrategy.cpp
  R600ClauseMergePass.cpp
  R600ControlFlowFinalizer.cpp
  R600EmitClauseMarkers.cpp
  R600ExpandSpecialInstrs.cpp
  R600FrameLowering.cpp
  R600InstrInfo.cpp
  R600ISelLowering.cpp
  R600MachineFunctionInfo.cpp
  R600MachineScheduler.cpp
  R600OptimizeVectorRegisters.cpp
  R600Packetizer.cpp
  R600RegisterInfo.cpp
  SIAnnotateControlFlow.cpp
  SIDebuggerInsertNops.cpp
  SIFixSGPRCopies.cpp
  SIFixVGPRCopies.cpp
  SIFixWWMLiveness.cpp
  SIFoldOperands.cpp
  SIFrameLowering.cpp
  SIInsertSkips.cpp
  SIInsertWaitcnts.cpp
  SIInsertWaits.cpp
  SIInstrInfo.cpp
  SIISelLowering.cpp
  SILoadStoreOptimizer.cpp
  SILowerControlFlow.cpp
  SILowerI1Copies.cpp
  SIMachineFunctionInfo.cpp
  SIMachineScheduler.cpp
  SIMemoryLegalizer.cpp
  SIOptimizeExecMasking.cpp
  SIOptimizeExecMaskingPreRA.cpp
  SIPeepholeSDWA.cpp
  SIRegisterInfo.cpp
  SIShrinkInstructions.cpp
  SIWholeQuadMode.cpp
  GCNILPSched.cpp
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
