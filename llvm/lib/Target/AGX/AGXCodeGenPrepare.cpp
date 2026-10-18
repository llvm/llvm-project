//===-- AGXCodeGenPrepare.cpp - AGX IR preparation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IR transformations that AGX code generation relies on:
//
//  * AGXAlwaysInline marks every non-kernel function always_inline (and
//    internal, so it disappears once inlined), since there are no calls,
//    and removes the unused ones.
//  * AGXCodeGenPrepare expands what the hardware has no instruction or
//    library for: integer division, and memory intrinsics that are not
//    small enough to be expanded by instruction selection. There is no
//    floating point environment, so constrained floating point intrinsics
//    become their ordinary operations. It also unifies the function exits,
//    which the control flow structurizer expects.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;

#define DEBUG_TYPE "agx-codegenprepare"

//===----------------------------------------------------------------------===//
// AGXAlwaysInline
//===----------------------------------------------------------------------===//

namespace {

class AGXAlwaysInline : public ModulePass {
public:
  static char ID;
  AGXAlwaysInline() : ModulePass(ID) {}

  StringRef getPassName() const override { return "AGX Always Inline"; }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // namespace

char AGXAlwaysInline::ID = 0;

INITIALIZE_PASS(AGXAlwaysInline, "agx-always-inline", "AGX Always Inline",
                false, false)

ModulePass *llvm::createAGXAlwaysInlinePass() { return new AGXAlwaysInline(); }

bool AGXAlwaysInline::runOnModule(Module &M) {
  // Modules without any agx_kernel function (for example tests of single
  // functions) compile every external function as an entry point.
  bool HasKernels = any_of(M, [](const Function &F) {
    return F.getCallingConv() == CallingConv::AGX_KERNEL;
  });

  bool Changed = false;
  SmallVector<Function *, 8> Dead;
  for (Function &F : M) {
    if (F.isDeclaration() || F.getCallingConv() == CallingConv::AGX_KERNEL)
      continue;
    if (!HasKernels && !F.hasLocalLinkage())
      continue;
    // With kernels in the module, other functions are only reachable from
    // them: internalize them so that they disappear once inlined, and drop
    // the unused ones (for example from linked device libraries).
    if (HasKernels && !F.hasLocalLinkage()) {
      F.setLinkage(GlobalValue::InternalLinkage);
      F.setVisibility(GlobalValue::DefaultVisibility);
      Changed = true;
    }
    if (F.use_empty()) {
      Dead.push_back(&F);
      continue;
    }
    if (F.hasFnAttribute(Attribute::NoInline))
      continue;
    F.addFnAttr(Attribute::AlwaysInline);
    Changed = true;
  }
  for (Function *F : Dead)
    F->eraseFromParent();
  Changed |= !Dead.empty();
  return Changed;
}

//===----------------------------------------------------------------------===//
// AGXCodeGenPrepare
//===----------------------------------------------------------------------===//

namespace {

class AGXCodeGenPrepare : public FunctionPass {
  bool expandDivisions(Function &F);
  bool expandMemIntrinsics(Function &F, const TargetTransformInfo &TTI);
  bool unifyExits(Function &F);
  bool lowerConstrainedFP(Function &F);

public:
  static char ID;
  AGXCodeGenPrepare() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX CodeGen Prepare"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
};

} // namespace

char AGXCodeGenPrepare::ID = 0;

INITIALIZE_PASS_BEGIN(AGXCodeGenPrepare, DEBUG_TYPE, "AGX CodeGen Prepare",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(AGXCodeGenPrepare, DEBUG_TYPE, "AGX CodeGen Prepare",
                    false, false)

FunctionPass *llvm::createAGXCodeGenPreparePass() {
  return new AGXCodeGenPrepare();
}

/// Division by a power of two is a shift, and 32-bit division by another
/// constant is a multiply-high; both are left to instruction selection.
static bool isCheapDivision(const BinaryOperator &BO) {
  auto *C = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!C)
    return false;
  if (C->getValue().isPowerOf2())
    return true;
  return BO.getType()->isIntegerTy(32);
}

bool AGXCodeGenPrepare::expandDivisions(Function &F) {
  SmallVector<BinaryOperator *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || !BO->getType()->isIntegerTy())
      continue;
    switch (BO->getOpcode()) {
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      if (!isCheapDivision(*BO) &&
          BO->getType()->getIntegerBitWidth() <= 64)
        Worklist.push_back(BO);
      break;
    default:
      break;
    }
  }

  for (BinaryOperator *BO : Worklist) {
    bool IsRem = BO->getOpcode() == Instruction::URem ||
                 BO->getOpcode() == Instruction::SRem;
    bool Is64 = BO->getType()->getIntegerBitWidth() > 32;
    if (IsRem)
      Is64 ? expandRemainderUpTo64Bits(BO) : expandRemainderUpTo32Bits(BO);
    else
      Is64 ? expandDivisionUpTo64Bits(BO) : expandDivisionUpTo32Bits(BO);
  }
  return !Worklist.empty();
}

bool AGXCodeGenPrepare::expandMemIntrinsics(Function &F,
                                            const TargetTransformInfo &TTI) {
  // Instruction selection expands memory intrinsics of a known, small size
  // into loads and stores; everything else becomes a loop.
  constexpr uint64_t MaxInlineSize = 128;
  SmallVector<MemIntrinsic *, 4> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *MI = dyn_cast<MemIntrinsic>(&I);
    if (!MI)
      continue;
    auto *Len = dyn_cast<ConstantInt>(MI->getLength());
    if (Len && Len->getZExtValue() <= MaxInlineSize)
      continue;
    Worklist.push_back(MI);
  }

  for (MemIntrinsic *MI : Worklist) {
    if (auto *Cpy = dyn_cast<MemCpyInst>(MI)) {
      expandMemCpyAsLoop(Cpy, TTI);
    } else if (auto *Move = dyn_cast<MemMoveInst>(MI)) {
      if (!expandMemMoveAsLoop(Move, TTI))
        continue;
    } else if (auto *Set = dyn_cast<MemSetInst>(MI)) {
      expandMemSetAsLoop(Set, TTI);
    } else {
      continue;
    }
    MI->eraseFromParent();
  }
  return !Worklist.empty();
}

bool AGXCodeGenPrepare::lowerConstrainedFP(Function &F) {
  SmallVector<ConstrainedFPIntrinsic *, 8> Worklist;
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<ConstrainedFPIntrinsic>(&I))
      Worklist.push_back(CI);

  for (ConstrainedFPIntrinsic *CI : Worklist) {
    IRBuilder<> B(CI);
    SmallVector<Value *, 4> Args(CI->args().begin(),
                                 CI->args().begin() +
                                     CI->getNonMetadataArgCount());
    Value *Res = nullptr;
    switch (CI->getIntrinsicID()) {
#define INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                        \
  case Intrinsic::INTRINSIC:                                                   \
    if (Instruction::isBinaryOp(Instruction::NAME))                            \
      Res = B.CreateBinOp(Instruction::BinaryOps(Instruction::NAME), Args[0],  \
                          Args[1]);                                            \
    else                                                                       \
      Res = B.CreateCast(Instruction::CastOps(Instruction::NAME), Args[0],     \
                         CI->getType());                                       \
    break;
#define CMP_INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC, DAGN)              \
  case Intrinsic::INTRINSIC:                                                   \
    Res = B.CreateFCmp(cast<ConstrainedFPCmpIntrinsic>(CI)->getPredicate(),    \
                       Args[0], Args[1]);                                      \
    break;
#define FUNCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                           \
  case Intrinsic::INTRINSIC: {                                                 \
    SmallVector<Type *, 4> ArgTys;                                             \
    for (Value *A : Args)                                                      \
      ArgTys.push_back(A->getType());                                          \
    SmallVector<Type *, 2> OverloadTys;                                        \
    FunctionType *FT = FunctionType::get(CI->getType(), ArgTys, false);        \
    if (Intrinsic::isSignatureValid(Intrinsic::NAME, FT, OverloadTys))         \
      Res = B.CreateCall(Intrinsic::getOrInsertDeclaration(                    \
                             F.getParent(), Intrinsic::NAME, OverloadTys),     \
                         Args);                                                \
    break;                                                                     \
  }
#include "llvm/IR/ConstrainedOps.def"
    default:
      break;
    }
    if (!Res)
      continue;
    Res->takeName(CI);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
  }
  return !Worklist.empty();
}

/// Replace the returns, and unreachable terminators, by branches to a single
/// return block.
bool AGXCodeGenPrepare::unifyExits(Function &F) {
  SmallVector<Instruction *, 4> Exits;
  for (BasicBlock &BB : F) {
    Instruction *T = BB.getTerminator();
    if (isa<ReturnInst>(T) || isa<UnreachableInst>(T))
      Exits.push_back(T);
  }
  if (Exits.size() <= 1)
    return false;
  if (!F.getReturnType()->isVoidTy())
    return false;

  BasicBlock *Ret = BasicBlock::Create(F.getContext(), "unified.return", &F);
  ReturnInst::Create(F.getContext(), Ret);
  for (Instruction *T : Exits) {
    UncondBrInst::Create(Ret, T->getIterator());
    T->eraseFromParent();
  }
  return true;
}

bool AGXCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  bool Changed = lowerConstrainedFP(F);
  Changed |= expandDivisions(F);
  Changed |= expandMemIntrinsics(F, TTI);
  Changed |= unifyExits(F);
  return Changed;
}
