add_llvm_component_group(AArch64 HAS_JIT)

set(LLVM_TARGET_DEFINITIONS AArch64.td)

llvm_tablegen(AArch64GenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(AArch64GenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(AArch64GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
llvm_tablegen(AArch64GenCallingConv.inc -gen-callingconv)
llvm_tablegen(AArch64GenDAGISel.inc -gen-dag-isel)
llvm_tablegen(AArch64GenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(AArch64GenFastISel.inc -gen-fast-isel)
llvm_tablegen(AArch64GenGlobalISel.inc -gen-global-isel -gisel-extended-llt)
llvm_tablegen(AArch64GenO0PreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AArch64O0PreLegalizerCombiner")
llvm_tablegen(AArch64GenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AArch64PreLegalizerCombiner")
llvm_tablegen(AArch64GenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AArch64PostLegalizerCombiner")
llvm_tablegen(AArch64GenPostLegalizeGILowering.inc -gen-global-isel-combiner
              -combiners="AArch64PostLegalizerLowering")
llvm_tablegen(AArch64GenInstrInfo.inc -gen-instr-info)
llvm_tablegen(AArch64GenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(AArch64GenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(AArch64GenRegisterBank.inc -gen-register-bank)
llvm_tablegen(AArch64GenRegisterInfo.inc -gen-register-info)
llvm_tablegen(AArch64GenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(AArch64GenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(AArch64GenSystemOperands.inc -gen-searchable-tables)
llvm_tablegen(AArch64GenExegesis.inc -gen-exegesis)

set(LLVM_TARGET_DEFINITIONS AArch64Options.td)
llvm_tablegen(AArch64Options.inc -gen-opt-parser-defs)

set(LLVM_TARGET_DEFINITIONS MCTargetDesc/AArch64MCOptions.td)
llvm_tablegen(AArch64MCOptions.inc -gen-opt-parser-defs)

add_public_tablegen_target(AArch64CommonTableGen)

add_llvm_target(AArch64CodeGen
  GISel/AArch64CallLowering.cpp
  GISel/AArch64GlobalISelUtils.cpp
  GISel/AArch64InstructionSelector.cpp
  GISel/AArch64LegalizerInfo.cpp
  GISel/AArch64O0PreLegalizerCombiner.cpp
  GISel/AArch64PreLegalizerCombiner.cpp
  GISel/AArch64PostLegalizerCombiner.cpp
  GISel/AArch64PostLegalizerLowering.cpp
  GISel/AArch64PostSelectOptimize.cpp
  GISel/AArch64RegisterBankInfo.cpp
  AArch64A57FPLoadBalancing.cpp
  AArch64CodeLayoutOpt.cpp
  AArch64AdvSIMDScalarPass.cpp
  AArch64Arm64ECCallLowering.cpp
  AArch64AsmPrinter.cpp
  AArch64BranchTargets.cpp
  AArch64CallingConvention.cpp
  AArch64CleanupLocalDynamicTLSPass.cpp
  AArch64CollectLOH.cpp
  AArch64CondBrTuning.cpp
  AArch64ConditionalCompares.cpp
  AArch64DeadRegisterDefinitionsPass.cpp
  AArch64ExpandImm.cpp
  AArch64ExpandPseudoInsts.cpp
  AArch64FalkorHWPFFix.cpp
  AArch64FastISel.cpp
  AArch64A53Fix835769.cpp
  AArch64FrameLowering.cpp
  AArch64CompressJumpTables.cpp
  AArch64ConditionOptimizer.cpp
  AArch64RedundantCopyElimination.cpp
  AArch64RedundantCondBranchPass.cpp
  AArch64ISelDAGToDAG.cpp
  AArch64ISelLowering.cpp
  AArch64InstrInfo.cpp
  AArch64LoadStoreOptimizer.cpp
  AArch64LowerHomogeneousPrologEpilog.cpp
  AArch64MachineFunctionInfo.cpp
  AArch64MachineScheduler.cpp
  AArch64MacroFusion.cpp
  AArch64MIPeepholeOpt.cpp
  AArch64MCInstLower.cpp
  AArch64PerfectShuffle.cpp
  AArch64PointerAuth.cpp
  AArch64PostCoalescerPass.cpp
  AArch64PromoteConstant.cpp
  AArch64PTrueCoalescing.cpp
  AArch64PBQPRegAlloc.cpp
  AArch64PredicateAsCounterLoopRewrites.cpp
  AArch64RegisterInfo.cpp
  AArch64SMEAttributes.cpp
  AArch64SLSHardening.cpp
  AArch64SelectionDAGInfo.cpp
  AArch64SpeculationHardening.cpp
  AArch64StackTagging.cpp
  AArch64StackTaggingPreRA.cpp
  AArch64StorePairSuppress.cpp
  AArch64Subtarget.cpp
  AArch64TargetMachine.cpp
  AArch64TargetObjectFile.cpp
  AArch64TargetTransformInfo.cpp
  SMEPeepholeOpt.cpp
  SVEShuffleOpts.cpp
  MachineSMEABIPass.cpp
  AArch64SRLTDefineSuperRegs.cpp
  AArch64SIMDInstrOpt.cpp
  AArch64PrologueEpilogue.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  AArch64Desc
  AArch64Info
  AArch64Utils
  Analysis
  AsmPrinter
  CFGuard
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  Option
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize

  ADD_TO_COMPONENT
  AArch64
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
