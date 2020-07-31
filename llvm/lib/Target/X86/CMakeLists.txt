set(LLVM_TARGET_DEFINITIONS X86.td)

tablegen(LLVM X86GenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM X86GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM X86GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
tablegen(LLVM X86GenCallingConv.inc -gen-callingconv)
tablegen(LLVM X86GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM X86GenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM X86GenEVEX2VEXTables.inc -gen-x86-EVEX2VEX-tables)
tablegen(LLVM X86GenExegesis.inc -gen-exegesis)
tablegen(LLVM X86GenFastISel.inc -gen-fast-isel)
tablegen(LLVM X86GenGlobalISel.inc -gen-global-isel)
tablegen(LLVM X86GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM X86GenRegisterBank.inc -gen-register-bank)
tablegen(LLVM X86GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM X86GenSubtargetInfo.inc -gen-subtarget)

if (X86_GEN_FOLD_TABLES)
  tablegen(LLVM X86GenFoldTables.inc -gen-x86-fold-tables)
endif()

add_public_tablegen_target(X86CommonTableGen)

set(sources
  X86AsmPrinter.cpp
  X86AvoidTrailingCall.cpp
  X86CallFrameOptimization.cpp
  X86CallingConv.cpp
  X86CallLowering.cpp
  X86CmovConversion.cpp
  X86DomainReassignment.cpp
  X86DiscriminateMemOps.cpp
  X86ExpandPseudo.cpp
  X86FastISel.cpp
  X86FixupBWInsts.cpp
  X86FixupLEAs.cpp
  X86AvoidStoreForwardingBlocks.cpp
  X86FixupSetCC.cpp
  X86FlagsCopyLowering.cpp
  X86FloatingPoint.cpp
  X86FrameLowering.cpp
  X86InstructionSelector.cpp
  X86ISelDAGToDAG.cpp
  X86ISelLowering.cpp
  X86IndirectBranchTracking.cpp
  X86IndirectThunks.cpp
  X86InterleavedAccess.cpp
  X86InsertPrefetch.cpp
  X86InstCombineIntrinsic.cpp
  X86InstrFMA3Info.cpp
  X86InstrFoldTables.cpp
  X86InstrInfo.cpp
  X86EvexToVex.cpp
  X86LegalizerInfo.cpp
  X86LoadValueInjectionLoadHardening.cpp
  X86LoadValueInjectionRetHardening.cpp
  X86MCInstLower.cpp
  X86MachineFunctionInfo.cpp
  X86MacroFusion.cpp
  X86OptimizeLEAs.cpp
  X86PadShortFunction.cpp
  X86PartialReduction.cpp
  X86RegisterBankInfo.cpp
  X86RegisterInfo.cpp
  X86SelectionDAGInfo.cpp
  X86ShuffleDecodeConstantPool.cpp
  X86SpeculativeLoadHardening.cpp
  X86SpeculativeExecutionSideEffectSuppression.cpp
  X86Subtarget.cpp
  X86TargetMachine.cpp
  X86TargetObjectFile.cpp
  X86TargetTransformInfo.cpp
  X86VZeroUpper.cpp
  X86WinAllocaExpander.cpp
  X86WinEHState.cpp
  X86InsertWait.cpp
  )

add_llvm_target(X86CodeGen ${sources})

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
