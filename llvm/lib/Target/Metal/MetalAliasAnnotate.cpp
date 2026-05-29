//===- MetalAliasAnnotate.cpp - AIR-style aliasing metadata ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalAliasAnnotate.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-alias-annotate"

namespace {

// Buffer-arg pointer = AS1 (device) or AS3 (threadgroup) pointer parameter.
static bool isBufferArg(const Argument &A) {
  if (!A.getType()->isPointerTy())
    return false;
  unsigned AS = A.getType()->getPointerAddressSpace();
  return AS == metal::AS::Device || AS == metal::AS::Threadgroup;
}

// Walk back through GEP / bitcast / addrspacecast / ptrtoint+inttoptr pairs.
// Return the unique Argument the pointer reaches, or nullptr if ambiguous.
static const Argument *reachingBufferArg(const Value *V,
                                         SmallPtrSetImpl<const Value *> &Seen) {
  if (!Seen.insert(V).second)
    return nullptr;
  if (auto *A = dyn_cast<Argument>(V))
    return isBufferArg(*A) ? A : nullptr;
  if (auto *GEP = dyn_cast<GEPOperator>(V))
    return reachingBufferArg(GEP->getPointerOperand(), Seen);
  if (auto *BC = dyn_cast<BitCastOperator>(V))
    return reachingBufferArg(BC->getOperand(0), Seen);
  if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V))
    return reachingBufferArg(ASC->getPointerOperand(), Seen);
  if (isa<LoadInst>(V))
    // Pointer loaded from memory: source unknown.
    return nullptr;
  if (auto *Sel = dyn_cast<SelectInst>(V)) {
    const Argument *T = reachingBufferArg(Sel->getTrueValue(), Seen);
    const Argument *F = reachingBufferArg(Sel->getFalseValue(), Seen);
    return (T && T == F) ? T : nullptr;
  }
  if (auto *PN = dyn_cast<PHINode>(V)) {
    const Argument *R = nullptr;
    for (unsigned I = 0; I < PN->getNumIncomingValues(); I++) {
      const Argument *In = reachingBufferArg(PN->getIncomingValue(I), Seen);
      if (!In)
        return nullptr;
      if (!R)
        R = In;
      else if (R != In)
        return nullptr;
    }
    return R;
  }
  // IntToPtr of a PtrToInt: peer through.
  if (auto *I2P = dyn_cast<IntToPtrInst>(V))
    if (auto *P2I = dyn_cast<PtrToIntInst>(I2P->getOperand(0)))
      return reachingBufferArg(P2I->getPointerOperand(), Seen);
  return nullptr;
}

static const Argument *reachingBufferArg(const Value *V) {
  SmallPtrSet<const Value *, 16> Seen;
  return reachingBufferArg(V, Seen);
}

static bool annotateFunction(Function &F) {
  // Collect buffer-arg parameters.
  SmallVector<Argument *, 8> BufArgs;
  for (auto &A : F.args())
    if (isBufferArg(A))
      BufArgs.push_back(&A);
  if (BufArgs.empty())
    return false;

  LLVMContext &Ctx = F.getContext();
  MDBuilder MDB(Ctx);

  // Per-kernel scope domain.
  std::string DomainName =
      ("air-alias-scopes(" + F.getName() + ")").str();
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain(DomainName);

  // Per-buffer-arg scope.
  SmallVector<MDNode *, 8> ArgScope(BufArgs.size(), nullptr);
  for (unsigned I = 0; I < BufArgs.size(); I++) {
    unsigned ArgNo = BufArgs[I]->getArgNo();
    std::string ScopeName =
        "air-alias-scope-arg(" + utostr(ArgNo) + ")";
    ArgScope[I] = MDB.createAnonymousAliasScope(Domain, ScopeName);
  }

  // Map Argument -> index in BufArgs.
  DenseMap<const Argument *, unsigned> ArgIdx;
  for (unsigned I = 0; I < BufArgs.size(); I++)
    ArgIdx[BufArgs[I]] = I;

  bool Changed = false;

  // Tag every load/store whose pointer reaches a single buffer arg.
  for (auto &BB : F) {
    for (auto &I : BB) {
      Value *Ptr = nullptr;
      if (auto *LI = dyn_cast<LoadInst>(&I))
        Ptr = LI->getPointerOperand();
      else if (auto *SI = dyn_cast<StoreInst>(&I))
        Ptr = SI->getPointerOperand();
      else
        continue;

      const Argument *A = reachingBufferArg(Ptr);
      if (!A)
        continue;
      auto It = ArgIdx.find(A);
      if (It == ArgIdx.end())
        continue;
      unsigned Idx = It->second;

      // alias.scope = {this arg's scope}
      SmallVector<Metadata *, 1> S = {ArgScope[Idx]};
      I.setMetadata(LLVMContext::MD_alias_scope, MDNode::get(Ctx, S));

      // noalias = {all other buffer args' scopes}
      if (BufArgs.size() > 1) {
        SmallVector<Metadata *, 8> N;
        for (unsigned J = 0; J < BufArgs.size(); J++)
          if (J != Idx)
            N.push_back(ArgScope[J]);
        I.setMetadata(LLVMContext::MD_noalias, MDNode::get(Ctx, N));
      }
      Changed = true;
    }
  }

  // Mirror Apple's parameter attribute on every buffer-arg pointer.
  for (auto *A : BufArgs) {
    if (A->hasAttribute("air-buffer-no-alias"))
      continue;
    A->addAttr(Attribute::get(Ctx, "air-buffer-no-alias"));
    Changed = true;
  }

  return Changed;
}

static bool runImpl(Module &M) {
  bool Changed = false;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    Changed |= annotateFunction(F);
  }
  return Changed;
}

} // namespace

PreservedAnalyses MetalAliasAnnotatePass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return runImpl(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool MetalAliasAnnotateLegacy::runOnModule(Module &M) { return runImpl(M); }

char MetalAliasAnnotateLegacy::ID = 0;

INITIALIZE_PASS(MetalAliasAnnotateLegacy, DEBUG_TYPE,
                "Metal AIR-style alias-metadata annotate", false, false)

ModulePass *llvm::createMetalAliasAnnotateLegacyPass() {
  return new MetalAliasAnnotateLegacy();
}
