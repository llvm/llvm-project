set(LLVM_TARGET_DEFINITIONS AMDGPU.td)

tablegen(LLVM AMDGPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AMDGPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AMDGPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AMDGPUGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AMDGPUGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM AMDGPUGenIntrinsics.inc -gen-tgt-intrinsic)
tablegen(LLVM AMDGPUGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM AMDGPUGenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM AMDGPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AMDGPUGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AMDGPUGenDisassemblerTables.inc -gen-disassembler)
add_public_tablegen_target(AMDGPUCommonTableGen)

# List of all GlobalISel files.
set(GLOBAL_ISEL_FILES
  AMDGPUCallLowering.cpp
  )

# Add GlobalISel files to the dependencies if the user wants to build it.
if(LLVM_BUILD_GLOBAL_ISEL)
  set(GLOBAL_ISEL_BUILD_FILES ${GLOBAL_ISEL_FILES})
else()
  set(GLOBAL_ISEL_BUILD_FILES"")
  set(LLVM_OPTIONAL_SOURCES LLVMGlobalISel ${GLOBAL_ISEL_FILES})
endif()


add_llvm_target(AMDGPUCodeGen
  AMDILCFGStructurizer.cpp
  AMDGPUAlwaysInlinePass.cpp
  AMDGPUAnnotateKernelFeatures.cpp
  AMDGPUAnnotateUniformValues.cpp
  AMDGPUAsmPrinter.cpp
  AMDGPUCodeGenPrepare.cpp
  AMDGPUFrameLowering.cpp
  AMDGPUTargetObjectFile.cpp
  AMDGPUIntrinsicInfo.cpp
  AMDGPUISelDAGToDAG.cpp
  AMDGPUMCInstLower.cpp
  AMDGPUMachineFunction.cpp
  AMDGPUOpenCLImageTypeLoweringPass.cpp
  AMDGPUSubtarget.cpp
  AMDGPUTargetMachine.cpp
  AMDGPUTargetTransformInfo.cpp
  AMDGPUISelLowering.cpp
  AMDGPUInstrInfo.cpp
  AMDGPUPromoteAlloca.cpp
  AMDGPURegisterInfo.cpp
  AMDConvertAtomicLibCalls.cpp
  AMDGPUOCL12Adapter.cpp
  AMDGPUPrintfRuntimeBinding.cpp
  GCNHazardRecognizer.cpp
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
  SIFixControlFlowLiveIntervals.cpp
  SIFixSGPRCopies.cpp
  SIFoldOperands.cpp
  SIFrameLowering.cpp
  SIInsertSkips.cpp
  SIInsertWaits.cpp
  SIInstrInfo.cpp
  SIISelLowering.cpp
  SILoadStoreOptimizer.cpp
  SILowerControlFlow.cpp
  SILowerI1Copies.cpp
  SIMachineFunctionInfo.cpp
  SIMachineScheduler.cpp
  SIMemoryLegalizer.cpp
  SIRegisterInfo.cpp
  SIShrinkInstructions.cpp
  SITypeRewriter.cpp
  SIWholeQuadMode.cpp
  ${GLOBAL_ISEL_BUILD_FILES}
  )

add_subdirectory(AsmParser)
add_subdirectory(InstPrinter)
add_subdirectory(Disassembler)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(Utils)
