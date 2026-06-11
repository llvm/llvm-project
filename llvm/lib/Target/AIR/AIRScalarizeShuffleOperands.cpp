//===- AIRScalarizeShuffleOperands.cpp - Scalarize shuffle inputs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRScalarizeShuffleOperands.h"
#include "AIR.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"

using namespace llvm;

#define DEBUG_TYPE "air-scalarize-shuffle-operands"

static bool isShuffleCall(const CallInst *CI) {
  if (const Function *F = CI->getCalledFunction())
    return F->getName().contains("simd_shuffle");
  return false;
}

// True when at least one `air.simd_shuffle*` operand is (transitively, through
// vector glue) derived from an extract of a vector SSA value — i.e. the lane
// value lives in a vector register at the shuffle.
static bool shuffleReadsVectorRegister(Function &F) {
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI || !isShuffleCall(CI))
      continue;
    // The shuffle value operand is operand 0.
    Value *Op = CI->getArgOperand(0);
    // Walk back through scalar FP ops / extracts; a shuffle whose operand is an
    // extractelement (or reaches one without leaving scalar arithmetic) was fed
    // from a vector register.
    SmallVector<Value *, 8> Work{Op};
    SmallPtrSet<Value *, 16> Seen;
    while (!Work.empty()) {
      Value *V = Work.pop_back_val();
      if (!Seen.insert(V).second)
        continue;
      if (isa<ExtractElementInst>(V))
        return true;
      auto *Inst = dyn_cast<Instruction>(V);
      if (!Inst)
        continue;
      // Only chase through value-preserving scalar arithmetic / casts and other
      // shuffles; do not cross loads, phis or vector builds (those break the
      // "lives in a vector register at the shuffle" chain we care about).
      if (isa<UnaryOperator>(Inst) || isa<BinaryOperator>(Inst) ||
          isa<CastInst>(Inst) || isa<SelectInst>(Inst) ||
          (isa<CallInst>(Inst) && isShuffleCall(cast<CallInst>(Inst))))
        for (Value *In : Inst->operands())
          Work.push_back(In);
    }
  }
  return false;
}

// True when the function performs vector floating-point *arithmetic* (not just
// vector memory load/store and the extract/insert glue around them). The SLP
// vectorizer (LLVM O1+) produces this in cross-lane reduce/scan kernels, and
// the resulting AIR is miscompiled by the AGX JIT the same way as the
// vector-fed shuffle pattern (wrong per-lane results). GEMM kernels never
// contain vector FP arithmetic — their only vectors are memory quads — so they
// are skipped and stay byte-identical.
static bool hasVectorComputeShape(Function &F) {
  auto isVecFP = [](Type *T) {
    auto *VT = dyn_cast<FixedVectorType>(T);
    return VT && VT->getElementType()->isFloatingPointTy();
  };
  for (Instruction &I : instructions(F)) {
    // Cross-lane vector lane permute (SLP repacks scan/segment data with these;
    // the AGX JIT miscompiles wide `shufflevector`s). GEMM never emits one.
    if (isa<ShuffleVectorInst>(I))
      return true;
    // Vector fadd/fsub/fmul/fdiv/frem/fneg.
    if ((isa<BinaryOperator>(I) || isa<UnaryOperator>(I)) &&
        I.getType()->isFPOrFPVectorTy() && I.getType()->isVectorTy())
      return true;
    // Vector fcmp.
    if (auto *FC = dyn_cast<FCmpInst>(&I))
      if (isVecFP(FC->getOperand(0)->getType()))
        return true;
    // Vector select over FP data.
    if (auto *Sel = dyn_cast<SelectInst>(&I))
      if (isVecFP(Sel->getType()))
        return true;
    // Vector min/max/fma intrinsics.
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *Callee = CI->getCalledFunction()) {
        StringRef N = Callee->getName();
        if ((N.contains("maxnum") || N.contains("minnum") ||
             N.contains("fmuladd") || N.contains("llvm.fma")) &&
            isVecFP(CI->getType()))
          return true;
      }
  }
  return false;
}

// A function needs the workaround when SLP has produced a cross-lane reduction
// or scan shape the AGX JIT miscompiles: either a vector-register value feeds a
// shuffle, or the function does vector FP arithmetic (the SLP reduction tree).
static bool needsScalarization(Function &F) {
  return shuffleReadsVectorRegister(F) || hasVectorComputeShape(F);
}

static bool scalarizeShuffleOperands(Module &M) {
  SmallVector<Function *, 4> Targets;
  for (Function &F : M)
    if (!F.isDeclaration() && needsScalarization(F))
      Targets.push_back(&F);
  if (Targets.empty())
    return false;

  // Default options: ScalarizeLoadStore = false, so vector memory accesses are
  // preserved; only the insert/extract/vector-arith glue is broken down. This
  // reproduces the scalar shuffle shape Apple's own frontend emits.
  PassBuilder PB;
  FunctionAnalysisManager FAM;
  PB.registerFunctionAnalyses(FAM);

  ScalarizerPass Scalarizer;
  bool Changed = false;
  for (Function *F : Targets) {
    Scalarizer.run(*F, FAM);
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses
AIRScalarizeShuffleOperandsPass::run(Module &M, ModuleAnalysisManager &AM) {
  return scalarizeShuffleOperands(M) ? PreservedAnalyses::none()
                                     : PreservedAnalyses::all();
}

bool AIRScalarizeShuffleOperandsLegacy::runOnModule(Module &M) {
  return scalarizeShuffleOperands(M);
}

char AIRScalarizeShuffleOperandsLegacy::ID = 0;

INITIALIZE_PASS(AIRScalarizeShuffleOperandsLegacy, DEBUG_TYPE,
                "AIR Scalarize Shuffle Operands", false, false)

ModulePass *llvm::createAIRScalarizeShuffleOperandsLegacyPass() {
  return new AIRScalarizeShuffleOperandsLegacy();
}
