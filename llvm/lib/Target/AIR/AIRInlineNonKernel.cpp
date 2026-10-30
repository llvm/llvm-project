//===- AIRInlineNonKernel.cpp - Inline non-kernel functions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIR has calls between functions, but no recursion (no call stack): Metal's
// compiler lays out every function's frame statically. The functions a kernel
// calls are inlined, except the large ones called from many places: inlining
// those multiplies the kernel's size (a Cycles kernel inlined a noise
// function 760 times and ran llc out of memory), so they stay functions
// (internal, called directly).
//
// A function stays a function when
//   size * (copies - 1) > air-keep-function-cost  and  size >= air-keep-function-min-size
// where size is its instruction count with the calls it inlines inlined, and
// copies is the number of copies inlining it would make (its call sites, each
// counted as many times as its caller is copied). air-keep-function-cost=0
// inlines everything, as AIR's consumers without calls need. Some functions
// are always inlined: hints (their consumer inlines or replaces them),
// functions with threadgroup variables (only a kernel has threadgroup
// memory), variadic ones and those with ABI-attributed parameters (byval,
// sret, ...) and those with pointers or values AIR has not (doubles, ...)
// in their signature.
//
//===----------------------------------------------------------------------===//

#include "AIRInlineNonKernel.h"
#include "AIR.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/AIR.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "air-inline-non-kernel"

static cl::opt<unsigned> KeepFunctionCost(
    "air-keep-function-cost",
    cl::desc("Keep a function the AIR target would inline as a function "
             "when inlining it would add more than this many instructions "
             "(its size times its copies after the first); 0 inlines every "
             "function"),
    cl::init(20000));

static cl::opt<unsigned> KeepFunctionMinSize(
    "air-keep-function-min-size",
    cl::desc("Inline every function smaller than this (instructions)"),
    cl::init(100));

namespace {
using Count = uint64_t;
constexpr Count Saturated = Count(1) << 48;

Count satAdd(Count A, Count B) { return std::min(A + B, Saturated); }
Count satMul(Count A, Count B) {
  if (A && B > Saturated / A)
    return Saturated;
  return std::min(A * B, Saturated);
}

bool isKernel(const Function &F, bool HaveKernelAttr) {
  return HaveKernelAttr ? F.hasFnAttribute("air-kernel") : false;
}

// Whether a kept function may take or return a value of type T: not a
// double (AIR has none; inlined code emulates them), an integer wider than 64
// bits (legalized where it is used), or a pointer. The AIR target follows a
// pointer's pointee type, address space and aliasing within a function (the
// typed pointers of AIR's bitcode, threadgroup variables' layout, alias
// scopes), not into the functions it is passed to.
bool isAIRValueType(Type *T) {
  if (auto *VT = dyn_cast<VectorType>(T))
    return isAIRValueType(VT->getElementType());
  if (auto *AT = dyn_cast<ArrayType>(T))
    return isAIRValueType(AT->getElementType());
  if (auto *ST = dyn_cast<StructType>(T))
    return all_of(ST->elements(), isAIRValueType);
  if (auto *IT = dyn_cast<IntegerType>(T))
    return IT->getBitWidth() <= 64;
  return T->isVoidTy() || T->isHalfTy() || T->isBFloatTy() || T->isFloatTy();
}

// Whether a constant refers to a threadgroup variable.
bool refersToThreadgroup(const Value *V,
                         SmallPtrSetImpl<const Constant *> &Seen) {
  auto *C = dyn_cast<Constant>(V);
  if (!C || isa<Function>(C) || !Seen.insert(C).second)
    return false;
  if (auto *GV = dyn_cast<GlobalVariable>(C))
    return GV->getAddressSpace() == air::AS::Threadgroup;
  return any_of(C->operands(), [&](const Value *Op) {
    return refersToThreadgroup(Op, Seen);
  });
}

bool usesThreadgroupVariable(const Function &F) {
  SmallPtrSet<const Constant *, 16> Seen;
  for (const Instruction &I : instructions(F))
    for (const Value *Op : I.operands())
      if (refersToThreadgroup(Op, Seen))
        return true;
  return false;
}

// Whether F may stay a function, as far as its own body and signature go.
bool mayKeep(const Function &F) {
  if (F.isVarArg() || F.hasFnAttribute(air::HintAttr) ||
      F.getName().starts_with(air::HintPrefix))
    return false;
  if (!isAIRValueType(F.getReturnType()) ||
      !all_of(F.getFunctionType()->params(), isAIRValueType))
    return false;
  for (const Argument &A : F.args())
    if (A.hasByValAttr() || A.hasByRefAttr() || A.hasStructRetAttr() ||
        A.hasInAllocaAttr() || A.hasPreallocatedAttr() ||
        A.hasSwiftSelfAttr() || A.hasSwiftErrorAttr())
      return false;
  // Called directly, and only so.
  for (const User *U : F.users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || CI->getCalledOperand() != &F || CI->isMustTailCall() ||
        CI->getFunctionType() != F.getFunctionType())
      return false;
  }
  return true;
}

unsigned instructionCount(const Function &F) {
  unsigned N = 0;
  for (const BasicBlock &BB : F)
    N += BB.size();
  return N;
}
} // namespace

static bool inlineNonKernel(Module &M) {
  // A recursive function cannot be inlined away (the loop below would inline
  // it forever, until memory runs out), and AIR has no call stack.
  CallGraph CG(M);
  SmallVector<Function *, 16> PostOrder; // callees before their callers
  for (scc_iterator<CallGraph *> I = scc_begin(&CG); !I.isAtEnd(); ++I) {
    for (CallGraphNode *N : *I)
      if (Function *F = N->getFunction(); F && !F->isDeclaration()) {
        if (I.hasCycle())
          reportFatalUsageError(
              ("AIR has no call stack: recursive function '" + F->getName() +
               "' cannot be compiled")
                  .str()
                  .c_str());
        PostOrder.push_back(F);
      }
  }

  // When the frontend marked the launchable entries ("air-kernel"), the other
  // functions nothing calls are dead (the mid-end optimizer may inline a
  // helper's only call site itself, leaving it uncalled); without marks they
  // are the kernels.
  bool HaveKernelAttr = any_of(M, [](const Function &F) {
    return !F.isDeclaration() && F.hasFnAttribute("air-kernel");
  });

  // The direct calls of each defined function, by caller.
  DenseMap<Function *, SmallVector<CallInst *, 4>> CallsOf;
  for (Function *F : PostOrder)
    for (Instruction &I : instructions(*F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *Callee = CI->getCalledFunction();
            Callee && !Callee->isDeclaration())
          CallsOf[Callee].push_back(CI);
  auto IsRoot = [&](Function *F) {
    return isKernel(*F, HaveKernelAttr) ||
           (!HaveKernelAttr && !CallsOf.count(F));
  };

  // Copies: how many copies of each function inlining would make, given what
  // its callers become (Keep). Callers first.
  DenseMap<Function *, bool> Keep;
  auto Copies = [&](DenseMap<Function *, Count> &C) {
    C.clear();
    for (Function *F : reverse(PostOrder)) {
      if (IsRoot(F)) {
        C[F] = 1;
        continue;
      }
      Count N = 0;
      for (CallInst *CI : CallsOf.lookup(F)) {
        Function *Caller = CI->getFunction();
        N = satAdd(N, IsRoot(Caller) || Keep.lookup(Caller) ? 1
                                                            : C.lookup(Caller));
      }
      C[F] = N;
    }
  };

  // Callees first: a function's size is its own plus that of the callees it
  // inlines; it stays a function when inlining it everywhere (its callers
  // inlined too, for now) would cost too much.
  DenseMap<Function *, Count> Size, FullCopies;
  DenseMap<Function *, bool> Threadgroup;
  Copies(FullCopies);
  for (Function *F : PostOrder) {
    Count S = instructionCount(*F);
    bool TG = usesThreadgroupVariable(*F);
    for (Instruction &I : instructions(*F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *Callee = CI->getCalledFunction();
            Callee && !Callee->isDeclaration() && !Keep.lookup(Callee)) {
          S = satAdd(S, Size.lookup(Callee));
          TG |= Threadgroup.lookup(Callee);
        }
    Size[F] = S;
    Threadgroup[F] = TG;
    Count N = FullCopies.lookup(F);
    Keep[F] = KeepFunctionCost && !IsRoot(F) && N > 1 && !TG && mayKeep(*F) &&
              S >= KeepFunctionMinSize &&
              satMul(S, N - 1) > Count(KeepFunctionCost);
  }

  // A function kept for copies its kept callers do not make (one called from
  // a single place in a kept function) is inlined after all. Callers first,
  // so each sees its callers' final choice.
  DenseMap<Function *, Count> Real;
  for (Function *F : reverse(PostOrder)) {
    if (!Keep.lookup(F))
      continue;
    Copies(Real);
    Count N = Real.lookup(F);
    if (N <= 1 || satMul(Size[F], N - 1) <= Count(KeepFunctionCost))
      Keep[F] = false;
  }

  LLVM_DEBUG({
    for (Function *F : PostOrder)
      if (Keep.lookup(F))
        dbgs() << "air-inline-non-kernel: keeping " << F->getName()
               << " (size " << Size[F] << ", copies " << FullCopies[F]
               << ")\n";
  });

  // Inline, callees first, so that each function inlined has its own inlined
  // callees in it already.
  bool Changed = false;
  // (CallsOf's calls are inlined from here on; only its keys are used.)
  for (Function *F : PostOrder) {
    // (Not into the dead functions erased below.)
    if (!IsRoot(F) && !FullCopies.lookup(F))
      continue;
    SmallVector<CallInst *, 16> ToInline;
    for (Instruction &I : instructions(*F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *Callee = CI->getCalledFunction();
            Callee && !Callee->isDeclaration() && !Keep.lookup(Callee))
          ToInline.push_back(CI);
    for (CallInst *CI : ToInline) {
      Function *Callee = CI->getCalledFunction();
      // Strip noinline so InlineFunction succeeds even on frontend-marked
      // functions.
      Callee->removeFnAttr(Attribute::NoInline);
      InlineFunctionInfo IFI;
      InlineResult Res = InlineFunction(*CI, IFI);
      if (!Res.isSuccess())
        reportFatalUsageError(("AIR: cannot inline '" + Callee->getName() +
                               "' into '" + F->getName() +
                               "': " + Res.getFailureReason())
                                  .str()
                                  .c_str());
      Changed = true;
    }
  }

  // The functions kept are called directly from this module only.
  for (Function *F : PostOrder)
    if (Keep.lookup(F)) {
      F->setLinkage(GlobalValue::InternalLinkage);
      F->setVisibility(GlobalValue::DefaultVisibility);
      F->setDSOLocal(true);
      Changed = true;
    }

  // Erase the functions inlined everywhere, and (when kernels are marked)
  // the other uncalled ones. Callers first: erasing one may leave its
  // callees uncalled.
  bool Erased = true;
  while (Erased) {
    Erased = false;
    SmallVector<Function *, 8> Dead;
    for (Function &F : M)
      if (!F.isDeclaration() && F.use_empty() && !isKernel(F, HaveKernelAttr) &&
          (HaveKernelAttr || CallsOf.count(&F)))
        Dead.push_back(&F);
    for (Function *F : Dead) {
      F->eraseFromParent();
      Erased = Changed = true;
    }
  }

  return Changed;
}

PreservedAnalyses AIRInlineNonKernelPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  return inlineNonKernel(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

bool AIRInlineNonKernelLegacy::runOnModule(Module &M) {
  return inlineNonKernel(M);
}

char AIRInlineNonKernelLegacy::ID = 0;

INITIALIZE_PASS(AIRInlineNonKernelLegacy, DEBUG_TYPE,
                "AIR Inline Non-Kernel Functions", false, false)

ModulePass *llvm::createAIRInlineNonKernelLegacyPass() {
  return new AIRInlineNonKernelLegacy();
}
