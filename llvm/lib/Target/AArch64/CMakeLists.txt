set(LLVM_TARGET_DEFINITIONS AArch64.td)

tablegen(LLVM AArch64GenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AArch64GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AArch64GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
tablegen(LLVM AArch64GenCallingConv.inc -gen-callingconv)
tablegen(LLVM AArch64GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AArch64GenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM AArch64GenFastISel.inc -gen-fast-isel)
tablegen(LLVM AArch64GenGlobalISel.inc -gen-global-isel)
tablegen(LLVM AArch64GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AArch64GenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM AArch64GenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM AArch64GenRegisterBank.inc -gen-register-bank)
tablegen(LLVM AArch64GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AArch64GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM AArch64GenSystemOperands.inc -gen-searchable-tables)
tablegen(LLVM AArch64GenExegesis.inc -gen-exegesis)

add_public_tablegen_target(AArch64CommonTableGen)

add_llvm_target(AArch64CodeGen
  AArch64A57FPLoadBalancing.cpp
  AArch64AdvSIMDScalarPass.cpp
  AArch64AsmPrinter.cpp
  AArch64BranchTargets.cpp
  AArch64CallingConvention.cpp
  AArch64CallLowering.cpp
  AArch64CleanupLocalDynamicTLSPass.cpp
  AArch64CollectLOH.cpp
  AArch64CondBrTuning.cpp
  AArch64ConditionalCompares.cpp
  AArch64DeadRegisterDefinitionsPass.cpp
  AArch64ExpandPseudoInsts.cpp
  AArch64FalkorHWPFFix.cpp
  AArch64FastISel.cpp
  AArch64A53Fix835769.cpp
  AArch64FrameLowering.cpp
  AArch64CompressJumpTables.cpp
  AArch64ConditionOptimizer.cpp
  AArch64RedundantCopyElimination.cpp
  AArch64ISelDAGToDAG.cpp
  AArch64ISelLowering.cpp
  AArch64InstrInfo.cpp
  AArch64InstructionSelector.cpp
  AArch64LegalizerInfo.cpp
  AArch64LoadStoreOptimizer.cpp
  AArch64MacroFusion.cpp
  AArch64MCInstLower.cpp
  AArch64PreLegalizerCombiner.cpp
  AArch64PromoteConstant.cpp
  AArch64PBQPRegAlloc.cpp
  AArch64RegisterBankInfo.cpp
  AArch64RegisterInfo.cpp
  AArch64SelectionDAGInfo.cpp
  AArch64SpeculationHardening.cpp
  AArch64StorePairSuppress.cpp
  AArch64Subtarget.cpp
  AArch64TargetMachine.cpp
  AArch64TargetObjectFile.cpp
  AArch64TargetTransformInfo.cpp
  AArch64SIMDInstrOpt.cpp

  DEPENDS
  intrinsics_gen
)

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
