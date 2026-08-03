//===-- PISAPropagateNullPointers.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass ensures null pointers remain null when cast between address
// spaces, handling both direct addrspacecast instructions and constant
// expressions.
//
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "PISASubtarget.h"
#include "PISATargetMachine.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PISAAddrSpace.h"

using namespace llvm;
using namespace llvm::PISA;
using namespace llvm::PISAAS;

#define DEBUG_TYPE "pisa-propagate-null-pointers"
#define DEBUG_NAME "PISA propagate null pointers"

namespace {

class PISAPropagateNullPointers : public ModulePass {
public:
  static char ID;

  PISAPropagateNullPointers() : ModulePass(ID) {}
  StringRef getPassName() const override { return DEBUG_NAME; }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    ModulePass::getAnalysisUsage(AU);
  }
};

} // namespace

char PISAPropagateNullPointers::ID = 0;
INITIALIZE_PASS(PISAPropagateNullPointers, DEBUG_TYPE, DEBUG_NAME, false, false)

// Creates a null pointer for the specified address space without using
// ConstantPointerNull::get, as LLVM assumes null pointers have a zero-bit
// representation regardless of address space which can lead to incorrect code.
static Constant *createNullPtr(PointerType *PtrTy, const DataLayout &DL) {
  unsigned AS = PtrTy->getAddressSpace();
  unsigned BitSize = DL.getPointerSizeInBits(AS);
  return ConstantExpr::getIntToPtr(
      ConstantInt::get(Type::getIntNTy(PtrTy->getContext(), BitSize),
                       PISATargetMachine::getNullPointerValue(AS), true),
      PtrTy);
}

static bool isPtrKnownNonNull(const Value *Src, const DataLayout &DL) {
  if (isa<GlobalValue, AllocaInst>(Src))
    return true;

  if (const auto *Arg = dyn_cast<Argument>(Src)) {
    if (Arg->hasNonNullAttr())
      return true;
    if (Arg->getParent()->getCallingConv() == CallingConv::PISA_KERNEL) {
      unsigned AS = Arg->getType()->getPointerAddressSpace();
      if (AS == static_cast<unsigned>(AddressSpace::PRIVATE) ||
          AS == static_cast<unsigned>(AddressSpace::SHARED))
        return true;
    }
  }

  return isKnownNonZero(Src, DL);
}

// Only casts between generic and shared/private address spaces are processed.
// Casts between global and generic address spaces are skipped since they do not
// change the pointer's bit representation.
static bool isCandidate(const AddrSpaceCastInst &ASC) {
  if (ASC.getType()->isVectorTy())
    return false;

  const unsigned SrcAS = ASC.getSrcAddressSpace();
  const unsigned DstAS = ASC.getDestAddressSpace();

  constexpr unsigned PrivateAS = static_cast<unsigned>(AddressSpace::PRIVATE);
  constexpr unsigned SharedAS = static_cast<unsigned>(AddressSpace::SHARED);
  constexpr unsigned GenericAS = static_cast<unsigned>(AddressSpace::GENERIC);

  if (SrcAS == GenericAS && (DstAS == PrivateAS || DstAS == SharedAS))
    return true;
  if ((SrcAS == PrivateAS || SrcAS == SharedAS) && DstAS == GenericAS)
    return true;
  return false;
}

static bool processASC(AddrSpaceCastInst &ASC, const DataLayout &DL) {
  if (!isCandidate(ASC))
    return false;

  auto *Src = ASC.getPointerOperand();
  SmallVector<const Value *, 4> WorkList;
  getUnderlyingObjects(Src, WorkList);
  if (all_of(WorkList,
             [&DL](const Value *V) { return isPtrKnownNonNull(V, DL); }))
    return false;

  auto *SrcNull = createNullPtr(cast<PointerType>(Src->getType()), DL);
  auto *DstNull = createNullPtr(cast<PointerType>(ASC.getType()), DL);

  IRBuilder<> IRB(&ASC);
  auto *ASCCopy = IRB.CreateAddrSpaceCast(Src, ASC.getType(), ASC.getName());
  auto *IsNonNull = IRB.CreateICmpNE(Src, SrcNull);
  auto *Select = IRB.CreateSelect(IsNonNull, ASCCopy, DstNull);
  ASC.replaceAllUsesWith(Select);
  ASC.eraseFromParent();
  return true;
}

// Replaces Clang-generated constant expression casts from generic null pointers
// to shared/private address spaces with inttoptr expressions. Only casts from
// generic to shared/private address spaces are processed.
static bool updateConstExprCasts(LLVMContext &Ctx, const DataLayout &DL) {
  auto *NullGeneric = ConstantPointerNull::get(
      PointerType::get(Ctx, static_cast<unsigned>(AddressSpace::GENERIC)));
  auto *NullPrivate = ConstantPointerNull::get(
      PointerType::get(Ctx, static_cast<unsigned>(AddressSpace::PRIVATE)));
  auto *NullShared = ConstantPointerNull::get(
      PointerType::get(Ctx, static_cast<unsigned>(AddressSpace::SHARED)));

  auto *GenericToPrivateCast =
      ConstantExpr::getAddrSpaceCast(NullGeneric, NullPrivate->getType());
  auto *GenericToSharedCast =
      ConstantExpr::getAddrSpaceCast(NullGeneric, NullShared->getType());

  bool Changed =
      !GenericToPrivateCast->use_empty() || !GenericToSharedCast->use_empty();
  GenericToPrivateCast->replaceAllUsesWith(
      createNullPtr(cast<PointerType>(NullPrivate->getType()), DL));
  GenericToSharedCast->replaceAllUsesWith(
      createNullPtr(cast<PointerType>(NullShared->getType()), DL));
  return Changed;
}

bool PISAPropagateNullPointers::runOnModule(Module &M) {
  bool Changed = false;
  for (auto &F : M)
    for (auto &I : make_early_inc_range(instructions(F)))
      if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I))
        Changed |= processASC(*ASC, M.getDataLayout());

  Changed |= updateConstExprCasts(M.getContext(), M.getDataLayout());
  return Changed;
}

ModulePass *llvm::createPISAPropagateNullPointersPass() {
  return new PISAPropagateNullPointers();
}
