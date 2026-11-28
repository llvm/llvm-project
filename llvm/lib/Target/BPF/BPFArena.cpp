//===-- BPFArena.cpp - What the arena passes have in common ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPFArena.h"
#include "BPF.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

/// An access to arena through NULL is not an error: it reads 0, and programs
/// that mark their pointers rely on that. LLVM takes it for defined in address
/// space 1 only. This runs in front of the optimizer.
PreservedAnalyses BPFArenaNullPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  F.addFnAttr(Attribute::NullPointerIsValid);
  return PreservedAnalyses::all();
}

bool llvm::isBPFArenaGlobal(const GlobalVariable &GV) {
  return GV.getAddressSpace() == BPFArenaAS ||
         GV.getSection() == ".addr_space.1" ||
         GV.getSection() == ".bss.addr_space.1";
}

bool llvm::isBPFKernelFunc(const CallBase &CB) {
  const Value *Callee = CB.getCalledOperand();

  if (CB.isInlineAsm() || isa<IntrinsicInst>(CB))
    return false;
  // A declaration without the section is a function in another file of
  // the program, or memcpy() and the like.
  if (const auto *F = dyn_cast<Function>(Callee))
    return F->isDeclaration() && F->getSection() == ".ksyms";
  // The number of a helper.
  return isa<Constant>(Callee);
}

const Argument *llvm::getBPFSourceParam(const Function &F, unsigned N) {
  if (F.hasParamAttribute(0, Attribute::StructRet))
    ++N;
  return N < F.arg_size() ? F.getArg(N) : nullptr;
}

/// __arg_nonnull, __arg_trusted and the like: the kernel verifies a function
/// that is not static with what they say.
static bool isDescribed(const DILocalVariable &DV) {
  DINodeArray Annotations = DV.getAnnotations();

  if (!Annotations)
    return false;
  return any_of(Annotations->operands(), [](const MDOperand &Op) {
    const auto *MD = cast<MDNode>(Op);
    StringRef Tag = cast<MDString>(MD->getOperand(1))->getString();

    return cast<MDString>(MD->getOperand(0))->getString() == "btf_decl_tag" &&
           Tag.starts_with("arg:") && Tag != "arg:arena";
  });
}

BPFKernelMem::BPFKernelMem(const Module &M) {
  for (const Function &F : M) {
    const DISubprogram *SP = F.getSubprogram();

    if (F.isDeclaration() || F.hasLocalLinkage() || !SP)
      continue;
    for (const MDNode *N : SP->getRetainedNodes())
      if (const auto *DV = dyn_cast<DILocalVariable>(N))
        if (DV->getArg() && isDescribed(*DV))
          if (const Argument *A = getBPFSourceParam(F, DV->getArg() - 1))
            Params.insert(A);
  }
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      for (const Argument &A : F.args()) {
        Visited Seen, SeenWeak;

        if (!A.getType()->isPointerTy() || Params.contains(&A))
          continue;
        // Other files call a function that is not static, and there its
        // parameters are in arena. Only what is plain to see changes that.
        Seen.Mode = F.hasLocalLinkage() ? Visited::Pointers : Visited::Direct;
        if (isPassed(&A, Seen))
          Changed |= Params.insert(&A).second;
        else if (F.hasLocalLinkage() && isPassed(&A, SeenWeak))
          Changed |= WeakParams.insert(&A).second;
      }
    }
  }
}

bool BPFKernelMem::isKernelUse(const Use &U, Visited &Seen) const {
  const User *Usr = U.getUser();

  if (const auto *CB = dyn_cast<CallBase>(Usr)) {
    if (!CB->isArgOperand(&U))
      return false;
    // A parameter that wants arena has another type, so the address gets
    // there through addrspacecast, which is not followed.
    if (isBPFKernelFunc(*CB))
      return true;
    // Asm without text is a barrier for the compiler. It accesses nothing.
    if (const auto *Asm = dyn_cast<InlineAsm>(CB->getCalledOperand()))
      return !Asm->getAsmString().empty();

    const Function *F = CB->getCalledFunction();
    unsigned N = CB->getArgOperandNo(&U);

    if (!F || N >= F->arg_size())
      return false;
    return Params.contains(F->getArg(N)) ||
           (Seen.Mode == Visited::Weak && WeakParams.contains(F->getArg(N)));
  }
  if (const auto *GEP = dyn_cast<GEPOperator>(Usr))
    return (Seen.Mode != Visited::Direct || GEP->hasAllConstantIndices()) &&
           isPassed(GEP, Seen);
  if (Seen.Mode == Visited::Direct)
    return false;
  if (isa<BitCastOperator>(Usr) || isa<SelectInst, PHINode, FreezeInst>(Usr))
    return isPassed(Usr, Seen);
  if (Seen.Mode == Visited::Pointers)
    return false;
  // The address goes into memory that the kernel gets, or it is loaded from
  // there and gets to the kernel then.
  if (const auto *SI = dyn_cast<StoreInst>(Usr)) {
    SmallVector<const Value *, 4> Objs;

    if (U.getOperandNo() == StoreInst::getPointerOperandIndex())
      return false;
    getUnderlyingObjects(SI->getPointerOperand(), Objs);
    return any_of(Objs, [&](const Value *Obj) {
      if (const auto *A = dyn_cast<Argument>(Obj))
        return Params.contains(A) || WeakParams.contains(A) ||
               isContentPassed(A, Seen);
      return isa<AllocaInst, GlobalVariable>(Obj) &&
             (isPassed(Obj, Seen) || isContentPassed(Obj, Seen));
    });
  }
  if (isa<PtrToIntOperator>(Usr) || isa<IntToPtrInst>(Usr))
    return isPassed(Usr, Seen);
  return false;
}

bool BPFKernelMem::isPassed(const Value *Ptr, Visited &Seen) const {
  if (!Seen.Addrs.insert(Ptr).second)
    return false;
  return any_of(Ptr->uses(),
                [&](const Use &U) { return isKernelUse(U, Seen); });
}

/// What the program loads from \p Ptr gets to the kernel.
bool BPFKernelMem::isContentPassed(const Value *Ptr, Visited &Seen) const {
  if (!Seen.Holders.insert(Ptr).second)
    return false;
  return any_of(Ptr->uses(), [&](const Use &U) {
    const User *Usr = U.getUser();

    if (isa<LoadInst>(Usr))
      return isPassed(Usr, Seen);
    if (isa<GEPOperator, BitCastOperator>(Usr) || isa<SelectInst, PHINode>(Usr))
      return isContentPassed(Usr, Seen);
    const auto *CB = dyn_cast<CallBase>(Usr);
    const Function *F = CB ? CB->getCalledFunction() : nullptr;

    return F && !F->isDeclaration() && CB->isArgOperand(&U) &&
           CB->getArgOperandNo(&U) < F->arg_size() &&
           isContentPassed(F->getArg(CB->getArgOperandNo(&U)), Seen);
  });
}

bool BPFKernelMem::isKernelUse(const Use &U) const {
  Visited Seen;

  return isKernelUse(U, Seen);
}

/// \p Obj is on the heap, or a global variable that the kernel does not get.
bool BPFKernelMem::isInArena(const Value *Obj, Visited &Seen) const {
  // A local says nothing: it follows what it holds.
  if (isa<AllocaInst>(Obj) || !Seen.Holders.insert(Obj).second)
    return false;
  if (isa<GlobalVariable>(Obj)) {
    Visited Holder;

    return !isPassed(Obj, Holder) && !isContentPassed(Obj, Holder);
  }
  const auto *A = dyn_cast<Argument>(Obj);
  if (!A)
    return true;
  if (!A->getParent()->hasLocalLinkage())
    return !Params.contains(A);
  return any_of(A->getParent()->users(), [&](const User *Usr) {
    SmallVector<const Value *, 4> Objs;
    const auto *CB = dyn_cast<CallBase>(Usr);

    if (!CB || CB->getCalledFunction() != A->getParent() ||
        A->getArgNo() >= CB->arg_size())
      return false;
    getUnderlyingObjects(CB->getArgOperand(A->getArgNo()), Objs);
    return any_of(Objs, [&](const Value *O) { return isInArena(O, Seen); });
  });
}

/// The address \p Ptr goes to the heap, to a global variable that the kernel
/// does not get, or to another file.
bool BPFKernelMem::goesToArena(const Value *Ptr, Visited &Seen) const {
  if (!Seen.Addrs.insert(Ptr).second)
    return false;
  return any_of(Ptr->uses(), [&](const Use &U) {
    const User *Usr = U.getUser();

    if (isa<GEPOperator, BitCastOperator, PtrToIntOperator>(Usr) ||
        isa<IntToPtrInst, SelectInst, PHINode, FreezeInst>(Usr))
      return goesToArena(Usr, Seen);
    if (const auto *SI = dyn_cast<StoreInst>(Usr)) {
      SmallVector<const Value *, 4> Objs;

      if (U.getOperandNo() == StoreInst::getPointerOperandIndex())
        return false;
      getUnderlyingObjects(SI->getPointerOperand(), Objs);
      return any_of(Objs,
                    [&](const Value *Obj) { return isInArena(Obj, Seen); });
    }
    const auto *CB = dyn_cast<CallBase>(Usr);
    if (!CB || !CB->isArgOperand(&U) || CB->isInlineAsm() ||
        isa<IntrinsicInst>(CB) || isBPFKernelFunc(*CB))
      return false;
    const Function *F = CB->getCalledFunction();
    unsigned N = CB->getArgOperandNo(&U);

    if (!F || N >= F->arg_size())
      return false;
    if (!F->isDeclaration() && F->hasLocalLinkage())
      return goesToArena(F->getArg(N), Seen);
    return !Params.contains(F->getArg(N));
  });
}

bool BPFKernelMem::isPassed(const Value *Ptr) const {
  Visited Strong, Weak, Arena;

  Strong.Mode = Visited::Pointers;
  return isPassed(Ptr, Strong) ||
         (isPassed(Ptr, Weak) && !goesToArena(Ptr, Arena));
}
