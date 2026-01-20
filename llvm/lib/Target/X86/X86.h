//===-- X86.h - Top-level interface for X86 representation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the x86
// target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86_H
#define LLVM_LIB_TARGET_X86_X86_H

#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/PassInfo.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class FunctionPass;
class InstructionSelector;
class PassRegistry;
class X86RegisterBankInfo;
class X86Subtarget;
class X86TargetMachine;

/// This pass converts a legalized DAG into a X86-specific DAG, ready for
/// instruction scheduling.
FunctionPass *createX86ISelDag(X86TargetMachine &TM, CodeGenOptLevel OptLevel);

/// This pass initializes a global base register for PIC on x86-32.
FunctionPass *createX86GlobalBaseRegPass();

/// This pass combines multiple accesses to local-dynamic TLS variables so that
/// the TLS base address for the module is only fetched once per execution path
/// through the function.
FunctionPass *createCleanupLocalDynamicTLSPass();

/// This function returns a pass which converts floating-point register
/// references and pseudo instructions into floating-point stack references and
/// physical instructions.
class X86FPStackifierPass : public PassInfoMixin<X86FPStackifierPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FPStackifierLegacyPass();

/// This pass inserts AVX vzeroupper instructions before each call to avoid
/// transition penalty between functions encoded with AVX and SSE.
FunctionPass *createX86IssueVZeroUpperPass();

/// This pass inserts ENDBR instructions before indirect jump/call
/// destinations as part of CET IBT mechanism.
FunctionPass *createX86IndirectBranchTrackingPass();

/// Return a pass that pads short functions with NOOPs.
/// This will prevent a stall when returning on the Atom.
FunctionPass *createX86PadShortFunctions();

/// Return a pass that selectively replaces certain instructions (like add,
/// sub, inc, dec, some shifts, and some multiplies) by equivalent LEA
/// instructions, in order to eliminate execution delays in some processors.
class X86FixupLEAsPass : public PassInfoMixin<X86FixupLEAsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FixupLEAsLegacyPass();

/// Return a pass that replaces equivalent slower instructions with faster
/// ones.
class X86FixupInstTuningPass : public PassInfoMixin<X86FixupInstTuningPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FixupInstTuningLegacyPass();

/// Return a pass that reduces the size of vector constant pool loads.
class X86FixupVectorConstantsPass
    : public PassInfoMixin<X86FixupInstTuningPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FixupVectorConstantsLegacyPass();

/// Return a pass that removes redundant LEA instructions and redundant address
/// recalculations.
class X86OptimizeLEAsPass : public PassInfoMixin<X86OptimizeLEAsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86OptimizeLEAsLegacyPass();

/// Return a pass that transforms setcc + movzx pairs into xor + setcc.
class X86FixupSetCCPass : public PassInfoMixin<X86FixupSetCCPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FixupSetCCLegacyPass();

/// Return a pass that avoids creating store forward block issues in the
/// hardware.
class X86AvoidStoreForwardingBlocksPass
    : public PassInfoMixin<X86AvoidStoreForwardingBlocksPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86AvoidStoreForwardingBlocksLegacyPass();

/// Return a pass that lowers EFLAGS copy pseudo instructions.
class X86FlagsCopyLoweringPass
    : public PassInfoMixin<X86FlagsCopyLoweringPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FlagsCopyLoweringLegacyPass();

/// Return a pass that expands DynAlloca pseudo-instructions.
class X86DynAllocaExpanderPass
    : public PassInfoMixin<X86DynAllocaExpanderPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86DynAllocaExpanderLegacyPass();

/// Return a pass that config the tile registers.
class X86TileConfigPass : public PassInfoMixin<X86TileConfigPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86TileConfigLegacyPass();

/// Return a pass that preconfig the tile registers before fast reg allocation.
class X86FastPreTileConfigPass
    : public PassInfoMixin<X86FastPreTileConfigPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FastPreTileConfigLegacyPass();

/// Return a pass that config the tile registers after fast reg allocation.
class X86FastTileConfigPass : public PassInfoMixin<X86FastTileConfigPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FastTileConfigLegacyPass();

/// Return a pass that insert pseudo tile config instruction.
class X86PreTileConfigPass : public PassInfoMixin<X86PreTileConfigPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86PreTileConfigLegacyPass();

/// Return a pass that lower the tile copy instruction.
class X86LowerTileCopyPass : public PassInfoMixin<X86LowerTileCopyPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86LowerTileCopyLegacyPass();

/// Return a pass that inserts int3 at the end of the function if it ends with a
/// CALL instruction. The pass does the same for each funclet as well. This
/// ensures that the open interval of function start and end PCs contains all
/// return addresses for the benefit of the Windows x64 unwinder.
class X86AvoidTrailingCallPass
    : public PassInfoMixin<X86AvoidTrailingCallPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

FunctionPass *createX86AvoidTrailingCallLegacyPass();

/// Return a pass that optimizes the code-size of x86 call sequences. This is
/// done by replacing esp-relative movs with pushes.
class X86CallFrameOptimizationPass
    : public PassInfoMixin<X86CallFrameOptimizationPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86CallFrameOptimizationLegacyPass();

/// Return an IR pass that inserts EH registration stack objects and explicit
/// EH state updates. This pass must run after EH preparation, which does
/// Windows-specific but architecture-neutral preparation.
FunctionPass *createX86WinEHStatePass();

/// Return a Machine IR pass that expands X86-specific pseudo
/// instructions into a sequence of actual instructions. This pass
/// must run after prologue/epilogue insertion and before lowering
/// the MachineInstr to MC.
class X86ExpandPseudoPass : public PassInfoMixin<X86ExpandPseudoPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86ExpandPseudoLegacyPass();

/// This pass converts X86 cmov instructions into branch when profitable.
class X86CmovConversionPass : public PassInfoMixin<X86CmovConversionPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86CmovConversionLegacyPass();

/// Return a Machine IR pass that selectively replaces
/// certain byte and word instructions by equivalent 32 bit instructions,
/// in order to eliminate partial register usage, false dependences on
/// the upper portions of registers, and to save code size.
class X86FixupBWInstsPass : public PassInfoMixin<X86FixupBWInstsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86FixupBWInstsLegacyPass();

/// Return a Machine IR pass that reassigns instruction chains from one domain
/// to another, when profitable.
class X86DomainReassignmentPass
    : public PassInfoMixin<X86DomainReassignmentPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86DomainReassignmentLegacyPass();

/// This pass compress instructions from EVEX space to legacy/VEX/EVEX space when
/// possible in order to reduce code size or facilitate HW decoding.
class X86CompressEVEXPass : public PassInfoMixin<X86CompressEVEXPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86CompressEVEXLegacyPass();

/// This pass creates the thunks for the retpoline feature.
FunctionPass *createX86IndirectThunksPass();

/// This pass replaces ret instructions with jmp's to __x86_return thunk.
class X86ReturnThunksPass : public PassInfoMixin<X86ReturnThunksPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86ReturnThunksLegacyPass();

/// This pass insert wait instruction after X87 instructions which could raise
/// fp exceptions when strict-fp enabled.
FunctionPass *createX86InsertX87waitPass();

/// This pass optimizes arithmetic based on knowledge that is only used by
/// a reduction sequence and is therefore safe to reassociate in interesting
/// ways.
class X86PartialReductionPass : public PassInfoMixin<X86PartialReductionPass> {
private:
  const X86TargetMachine *TM;

public:
  X86PartialReductionPass(const X86TargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

FunctionPass *createX86PartialReductionLegacyPass();

/// // Analyzes and emits pseudos to support Win x64 Unwind V2.
FunctionPass *createX86WinEHUnwindV2Pass();

/// The pass transforms load/store <256 x i32> to AMX load/store intrinsics
/// or split the data to two <128 x i32>.
class X86LowerAMXTypePass : public PassInfoMixin<X86LowerAMXTypePass> {
private:
  const TargetMachine *TM;

public:
  X86LowerAMXTypePass(const TargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
  static bool isRequired() { return true; }
};

FunctionPass *createX86LowerAMXTypeLegacyPass();

// Suppresses APX features for relocations for supporting older linkers.
class X86SuppressAPXForRelocationPass
    : public PassInfoMixin<X86SuppressAPXForRelocationPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86SuppressAPXForRelocationLegacyPass();

/// The pass transforms amx intrinsics to scalar operation if the function has
/// optnone attribute or it is O0.
class X86LowerAMXIntrinsicsPass
    : public PassInfoMixin<X86LowerAMXIntrinsicsPass> {
private:
  const TargetMachine *TM;

public:
  X86LowerAMXIntrinsicsPass(const TargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
  static bool isRequired() { return true; }
};

FunctionPass *createX86LowerAMXIntrinsicsLegacyPass();

InstructionSelector *createX86InstructionSelector(const X86TargetMachine &TM,
                                                  const X86Subtarget &,
                                                  const X86RegisterBankInfo &);

FunctionPass *createX86PreLegalizerCombiner();
FunctionPass *createX86LoadValueInjectionLoadHardeningPass();

class X86LoadValueInjectionRetHardeningPass
    : public PassInfoMixin<X86LoadValueInjectionRetHardeningPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86LoadValueInjectionRetHardeningLegacyPass();

class X86SpeculativeExecutionSideEffectSuppressionPass
    : public PassInfoMixin<X86SpeculativeExecutionSideEffectSuppressionPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86SpeculativeExecutionSideEffectSuppressionLegacyPass();

class X86SpeculativeLoadHardeningPass
    : public PassInfoMixin<X86SpeculativeLoadHardeningPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86SpeculativeLoadHardeningLegacyPass();

class X86ArgumentStackSlotPass
    : public PassInfoMixin<X86ArgumentStackSlotPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createX86ArgumentStackSlotLegacyPass();

void initializeCompressEVEXLegacyPass(PassRegistry &);
void initializeX86FixupBWInstLegacyPass(PassRegistry &);
void initializeFixupLEAsLegacyPass(PassRegistry &);
void initializeX86ArgumentStackSlotLegacyPass(PassRegistry &);
void initializeX86AsmPrinterPass(PassRegistry &);
void initializeX86FixupInstTuningLegacyPass(PassRegistry &);
void initializeX86FixupVectorConstantsLegacyPass(PassRegistry &);
void initializeWinEHStatePassPass(PassRegistry &);
void initializeX86AvoidSFBLegacyPass(PassRegistry &);
void initializeX86AvoidTrailingCallLegacyPassPass(PassRegistry &);
void initializeX86CallFrameOptimizationLegacyPass(PassRegistry &);
void initializeX86CmovConversionLegacyPass(PassRegistry &);
void initializeX86DAGToDAGISelLegacyPass(PassRegistry &);
void initializeX86DomainReassignmentLegacyPass(PassRegistry &);
void initializeX86DynAllocaExpanderLegacyPass(PassRegistry &);
void initializeX86ExecutionDomainFixPass(PassRegistry &);
void initializeX86ExpandPseudoLegacyPass(PassRegistry &);
void initializeX86FPStackifierLegacyPass(PassRegistry &);
void initializeX86FastPreTileConfigLegacyPass(PassRegistry &);
void initializeX86FastTileConfigLegacyPass(PassRegistry &);
void initializeX86FixupSetCCLegacyPass(PassRegistry &);
void initializeX86FlagsCopyLoweringLegacyPass(PassRegistry &);
void initializeX86LoadValueInjectionLoadHardeningPassPass(PassRegistry &);
void initializeX86LoadValueInjectionRetHardeningLegacyPass(PassRegistry &);
void initializeX86LowerAMXIntrinsicsLegacyPassPass(PassRegistry &);
void initializeX86LowerAMXTypeLegacyPassPass(PassRegistry &);
void initializeX86LowerTileCopyLegacyPass(PassRegistry &);
void initializeX86OptimizeLEAsLegacyPass(PassRegistry &);
void initializeX86PartialReductionLegacyPass(PassRegistry &);
void initializeX86PreTileConfigLegacyPass(PassRegistry &);
void initializeX86ReturnThunksLegacyPass(PassRegistry &);
void initializeX86SpeculativeExecutionSideEffectSuppressionLegacyPass(
    PassRegistry &);
void initializeX86SpeculativeLoadHardeningLegacyPass(PassRegistry &);
void initializeX86SuppressAPXForRelocationLegacyPass(PassRegistry &);
void initializeX86TileConfigLegacyPass(PassRegistry &);
void initializeX86WinEHUnwindV2Pass(PassRegistry &);
void initializeX86PreLegalizerCombinerPass(PassRegistry &);

namespace X86AS {
enum : unsigned {
  GS = 256,
  FS = 257,
  SS = 258,
  PTR32_SPTR = 270,
  PTR32_UPTR = 271,
  PTR64 = 272
};
} // End X86AS namespace

} // End llvm namespace

#endif
