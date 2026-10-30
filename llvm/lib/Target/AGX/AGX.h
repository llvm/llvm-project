//===-- AGX.h - Top-level interface for AGX ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry points of the AGX code generator passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGX_H
#define LLVM_LIB_TARGET_AGX_AGX_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AGXTargetMachine;
class ScheduleDAGMutation;
class ScheduleDAGMI;
struct MachineSchedContext;
class FunctionPass;
class IRBuilderBase;
class Module;
class ModulePass;
class PassRegistry;

bool agxExperimentalLeafCallsEnabled();

// IR passes.
ModulePass *createAGXAlwaysInlinePass();
ModulePass *createAGXGlobalDataArgsPass();
ModulePass *createAGXLowerAIRPass(const AGXTargetMachine *TM);
ModulePass *createAGXPromoteKernelArgsPass();
ModulePass *createAGXCheckedAccessesPass(const AGXTargetMachine *TM);
FunctionPass *createAGXCodeGenPreparePass();
FunctionPass *createAGXStrictFPPass();
FunctionPass *createAGXLowerFPEnvPass();
FunctionPass *createAGXSoftF64Pass();
FunctionPass *createAGXExpandWideAtomicsPass(const AGXTargetMachine *TM = nullptr);
FunctionPass *createAGXAnnotateControlFlowPass();
FunctionPass *createAGXSinkAddressesPass();
FunctionPass *createAGXThreadgroupAddressesPass();
FunctionPass *createAGXHoistGuardedLoadsPass();
FunctionPass *createAGXNarrowInductionsPass();
FunctionPass *createAGXSpecialRegisterRangesPass();
FunctionPass *createAGXVectorAccessesPass(bool BeforeVectorizer);
FunctionPass *createAGXLowerGenericAddressesPass();

// Replaces recursive functions by state machines over an explicit stack
// (AGXLowerRecursion.cpp).
bool lowerAGXRecursion(Module &M);

// Instruction selection.
FunctionPass *createAGXISelDag(AGXTargetMachine &TM);

// Machine passes.
FunctionPass *createAGXFoldOperandsPass();
FunctionPass *createAGXUniformConstantsPass();
FunctionPass *createAGXHoistFromElseFlowPass();
FunctionPass *createAGXLowerControlFlowPass();
FunctionPass *createAGXInsertWaitsPass();

// Machine scheduler: keeps runs of memory accesses of one kind together.
std::unique_ptr<ScheduleDAGMutation> createAGXMemClusterMutation();
/// The post-RA machine scheduler: AGXMemClusterMutation, and cluster members
/// kept together ahead of latency stalls.
ScheduleDAGMI *createAGXPostRAScheduler(MachineSchedContext *C);

void initializeAGXAlwaysInlinePass(PassRegistry &);
void initializeAGXLowerAIRLegacyPass(PassRegistry &);
void initializeAGXPromoteKernelArgsPass(PassRegistry &);
void initializeAGXCheckedAccessesPass(PassRegistry &);
void initializeAGXCodeGenPreparePass(PassRegistry &);
void initializeAGXStrictFPPass(PassRegistry &);
void initializeAGXLowerFPEnvPass(PassRegistry &);
void initializeAGXSoftF64Pass(PassRegistry &);
void initializeAGXExpandWideAtomicsPass(PassRegistry &);
void initializeAGXAnnotateControlFlowPass(PassRegistry &);
void initializeAGXSinkAddressesPass(PassRegistry &);
void initializeAGXThreadgroupAddressesPass(PassRegistry &);
void initializeAGXHoistGuardedLoadsPass(PassRegistry &);
void initializeAGXNarrowInductionsPass(PassRegistry &);
void initializeAGXSpecialRegisterRangesPassPass(PassRegistry &);
void initializeAGXVectorAccessesPass(PassRegistry &);
void initializeAGXLowerGenericAddressesPass(PassRegistry &);
void initializeAGXDAGToDAGISelLegacyPass(PassRegistry &);
void initializeAGXFoldOperandsPass(PassRegistry &);
void initializeAGXUniformConstantsPassPass(PassRegistry &);
void initializeAGXHoistFromElseFlowPass(PassRegistry &);
void initializeAGXLowerControlFlowPass(PassRegistry &);
void initializeAGXInsertWaitsPass(PassRegistry &);
void initializeAGXAsmPrinterPass(PassRegistry &);

class TargetMachine;

/// Compiles a module described by !air.kernel (AIR) as AGX kernels.
class AGXLowerAIRPass : public RequiredPassInfoMixin<AGXLowerAIRPass> {
  const TargetMachine &TM;

public:
  explicit AGXLowerAIRPass(const TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Writes a module of AGX kernels as the input of the AIR target's portable
/// passes: upstream AIR, with hints for what AIR does not have.
class AGXRaiseToAIRPass : public RequiredPassInfoMixin<AGXRaiseToAIRPass> {
  const TargetMachine *TM;

public:
  explicit AGXRaiseToAIRPass(const TargetMachine *TM = nullptr) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class Instruction;
class IRBuilderBase;
class MachineInstr;
class Type;
class Value;

namespace AGX {
/// Whether expandMath builds intrinsic \p ID of an operand of type \p Ty
/// (and with afn if \p Approx): tan, asin, acos, atan, atan2, sinh, cosh and
/// tanh of float, half or bfloat (or vectors), and the precise (not afn)
/// float sin, cos, sincos, exp, exp2, exp10, log, log2, log10 and pow.
bool canExpandMath(Intrinsic::ID ID, Type *Ty, bool Approx);
/// The math functions AGX has no instruction for, from the ones it has
/// (AGXMathExpansion.cpp): the intrinsic \p ID of \p Args, or AIR's \p Fn
/// (asinh, acosh, atanh, sinpi, cospi, tanpi) of \p X; nullptr for another.
Value *expandMath(IRBuilderBase &B, Intrinsic::ID ID, ArrayRef<Value *> Args);
Value *expandMath(IRBuilderBase &B, StringRef Fn, Value *X);

/// Whether a device memory access must reach memory shared with the host while
/// the dispatch runs: volatile and system-scope atomic loads and stores
/// (PTX ld/st.volatile, .sys). They are encoded coherent, loads after an f59d
/// and stores before an f533 (ptx-re-blockers B7).
bool isHostCoherentAccess(const Instruction &I);
bool isHostCoherentAccess(const MachineInstr &MI);
} // namespace AGX

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGX_H
