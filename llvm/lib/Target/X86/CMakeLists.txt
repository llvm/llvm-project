add_llvm_component_group(X86 HAS_JIT)

set(LLVM_TARGET_DEFINITIONS X86.td)

llvm_tablegen(X86GenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(X86GenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(X86GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
llvm_tablegen(X86GenCallingConv.inc -gen-callingconv)
llvm_tablegen(X86GenDAGISel.inc -gen-dag-isel)
llvm_tablegen(X86GenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(X86GenInstrMapping.inc -gen-x86-instr-mapping)
llvm_tablegen(X86GenExegesis.inc -gen-exegesis)
llvm_tablegen(X86GenFastISel.inc -gen-fast-isel)
llvm_tablegen(X86GenGlobalISel.inc -gen-global-isel)
llvm_tablegen(X86GenInstrInfo.inc -gen-instr-info
                                  -instr-info-expand-mi-operand-info=0)
llvm_tablegen(X86GenMnemonicTables.inc -gen-x86-mnemonic-tables -asmwriternum=1)
llvm_tablegen(X86GenRegisterBank.inc -gen-register-bank)
llvm_tablegen(X86GenRegisterInfo.inc -gen-register-info)
llvm_tablegen(X86GenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(X86GenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(X86GenFoldTables.inc -gen-x86-fold-tables -asmwriternum=1)
llvm_tablegen(X86GenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="X86PreLegalizerCombiner")
llvm_tablegen(X86GenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="X86PostLegalizerCombiner")

set(LLVM_TARGET_DEFINITIONS X86Options.td)
llvm_tablegen(X86Options.inc -gen-opt-parser-defs)

set(LLVM_TARGET_DEFINITIONS MCTargetDesc/X86MCOptions.td)
llvm_tablegen(X86MCOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(X86CommonTableGen)

set(sources
  X86ArgumentStackSlotRebase.cpp
  X86AsmPrinter.cpp
  X86AvoidTrailingCall.cpp
  X86CallFrameOptimization.cpp
  X86CallingConv.cpp
  X86CleanupLocalDynamicTLS.cpp
  X86CmovConversion.cpp
  X86CodeGenPassBuilder.cpp
  X86DomainReassignment.cpp
  X86GlobalBaseReg.cpp
  X86LowerTileCopy.cpp
  X86LowerAMXType.cpp
  X86LowerAMXIntrinsics.cpp
  X86TileConfig.cpp
  X86FastPreTileConfig.cpp
  X86FastTileConfig.cpp
  X86PreTileConfig.cpp
  X86ExpandPseudo.cpp
  X86FastISel.cpp
  X86FenceNonTemporalStores.cpp
  X86FixupBWInsts.cpp
  X86FixupLEAs.cpp
  X86FixupInstTuning.cpp
  X86FixupVectorConstants.cpp
  X86AvoidStoreForwardingBlocks.cpp
  X86DynAllocaExpander.cpp
  X86FixupSetCC.cpp
  X86FlagsCopyLowering.cpp
  X86FloatingPoint.cpp
  X86FrameLowering.cpp
  X86ISelDAGToDAG.cpp
  X86ISelLowering.cpp
  X86ISelLoweringCall.cpp
  X86IndirectBranchTracking.cpp
  X86IndirectThunks.cpp
  X86InsertVZeroUpper.cpp
  X86InsertX87Wait.cpp
  X86InterleavedAccess.cpp
  X86InstCombineIntrinsic.cpp
  X86InstrFMA3Info.cpp
  X86InstrFoldTables.cpp
  X86InstrInfo.cpp
  X86CompressEVEX.cpp
  X86LFIRewritePass.cpp
  X86LoadValueInjectionLoadHardening.cpp
  X86LoadValueInjectionRetHardening.cpp
  X86MCInstLower.cpp
  X86MachineFunctionInfo.cpp
  X86MacroFusion.cpp
  X86OptimizeLEAs.cpp
  X86PadShortFunction.cpp
  X86PartialReduction.cpp
  X86RegisterInfo.cpp
  X86ReturnThunks.cpp
  X86SelectionDAGInfo.cpp
  X86ShuffleDecodeConstantPool.cpp
  X86SpeculativeLoadHardening.cpp
  X86SpeculativeExecutionSideEffectSuppression.cpp
  X86Subtarget.cpp
  X86SuppressAPXForReloc.cpp
  X86TargetMachine.cpp
  X86TargetObjectFile.cpp
  X86TargetTransformInfo.cpp
  X86WinEHState.cpp
  X86WinEHUnwindV2.cpp
  X86WinEHUnwindV3.cpp
  GISel/X86CallLowering.cpp
  GISel/X86InstructionSelector.cpp
  GISel/X86LegalizerInfo.cpp
  GISel/X86PostLegalizerCombiner.cpp
  GISel/X86PreLegalizerCombiner.cpp
  GISel/X86RegisterBankInfo.cpp
  )

add_llvm_target(X86CodeGen ${sources}
  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CFGuard
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  Instrumentation
  MC
  ObjCARC
  Option
  Passes
  ProfileData
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  X86Desc
  X86Info

  ADD_TO_COMPONENT
  X86
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCA)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
