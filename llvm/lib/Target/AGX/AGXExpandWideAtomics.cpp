//===-- AGXExpandWideAtomics.cpp - 64- and 128-bit atomics for AGX --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The hardware atomics are 32-bit. Atomic operations on wider values (64-bit
// and 128-bit loads, stores, read-modify-write operations and compare and
// swaps) take a lock from a table of device memory locks, chosen by the
// address of the 16-byte block they access, and access the value with 32-bit
// atomics, between fences, while they hold it:
//
//   loop:                          ; per thread, until it has done its work
//     if (cmpxchg(lock, 0, 1) succeeds) {
//       fence; read; compute; write; fence
//       xchg(lock, 0)
//       done
//     }
//
// The critical section is inside the loop, so a thread that holds a lock
// releases it in the same iteration, before the other threads of its
// SIMD-group, which may wait for the same lock, try again.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/LowerAtomic.h"

using namespace llvm;

#define DEBUG_TYPE "agx-expand-wide-atomics"

static constexpr unsigned NumLocks = 1024;

namespace {

class AGXExpandWideAtomics : public FunctionPass {
public:
  static char ID;
  explicit AGXExpandWideAtomics(const AGXTargetMachine *TM = nullptr)
      : FunctionPass(ID), TM(TM) {}

  StringRef getPassName() const override { return "AGX Expand Wide Atomics"; }
  bool runOnFunction(Function &F) override;

private:
  const AGXTargetMachine *TM;
};

} // namespace

char AGXExpandWideAtomics::ID = 0;

INITIALIZE_PASS(AGXExpandWideAtomics, DEBUG_TYPE, "AGX Expand Wide Atomics",
                false, false)

FunctionPass *llvm::createAGXExpandWideAtomicsPass(const AGXTargetMachine *TM) {
  return new AGXExpandWideAtomics(TM);
}

static GlobalVariable *getLocks(Module &M) {
  const char *Name = "__agx_atomic_locks";
  if (GlobalVariable *G = M.getNamedGlobal(Name))
    return G;
  auto *Ty = ArrayType::get(Type::getInt32Ty(M.getContext()), NumLocks);
  return new GlobalVariable(M, Ty, false, GlobalValue::LinkOnceODRLinkage,
                            Constant::getNullValue(Ty), Name, nullptr,
                            GlobalValue::NotThreadLocal, /*AddressSpace=*/1);
}

/// The size in bits of the value of a wide atomic instruction, or 0.
static unsigned wideAtomicSize(const Instruction &I, const DataLayout &DL) {
  Type *Ty = nullptr;
  if (auto *LI = dyn_cast<LoadInst>(&I); LI && LI->isAtomic())
    Ty = LI->getType();
  else if (auto *SI = dyn_cast<StoreInst>(&I); SI && SI->isAtomic())
    Ty = SI->getValueOperand()->getType();
  else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
    Ty = RMW->getType();
  else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
    Ty = CX->getCompareOperand()->getType();
  if (!Ty)
    return 0;
  unsigned Bits = DL.getTypeSizeInBits(Ty);
  return Bits == 64 || Bits == 128 ? Bits : 0;
}

static Value *getPointer(Instruction &I) {
  if (auto *LI = dyn_cast<LoadInst>(&I))
    return LI->getPointerOperand();
  if (auto *SI = dyn_cast<StoreInst>(&I))
    return SI->getPointerOperand();
  if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
    return RMW->getPointerOperand();
  return cast<AtomicCmpXchgInst>(I).getPointerOperand();
}

/// V (of any type of Bits bits) as an integer, and back.
static Value *toInt(IRBuilderBase &B, Value *V, unsigned Bits) {
  Type *IntTy = B.getIntNTy(Bits);
  if (V->getType()->isPointerTy())
    return B.CreatePtrToInt(V, IntTy);
  return B.CreateBitCast(V, IntTy);
}

static Value *fromInt(IRBuilderBase &B, Value *V, Type *Ty) {
  if (Ty->isPointerTy())
    return B.CreateIntToPtr(V, Ty);
  return B.CreateBitCast(V, Ty);
}

static void expand(Instruction &I, unsigned Bits, GlobalVariable *Locks) {
  const DataLayout &DL = I.getDataLayout();
  LLVMContext &Ctx = I.getContext();
  Value *Ptr = getPointer(I);
  unsigned Words = Bits / 32;
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *ValTy = isa<StoreInst>(I) ? cast<StoreInst>(I).getValueOperand()->getType()
                : isa<AtomicCmpXchgInst>(I)
                    ? cast<AtomicCmpXchgInst>(I).getCompareOperand()->getType()
                    : I.getType();

  BasicBlock *Pre = I.getParent();
  BasicBlock *Exit = Pre->splitBasicBlock(I.getIterator(), "wide.atomic.end");
  Function *F = Pre->getParent();
  BasicBlock *Head = BasicBlock::Create(Ctx, "wide.atomic", F, Exit);
  BasicBlock *Crit = BasicBlock::Create(Ctx, "wide.atomic.locked", F, Exit);
  BasicBlock *Latch = BasicBlock::Create(Ctx, "wide.atomic.latch", F, Exit);
  Pre->getTerminator()->setSuccessor(0, Head);

  IRBuilder<> B(Pre->getTerminator());
  unsigned PtrBits = DL.getPointerSizeInBits(Ptr->getType()->getPointerAddressSpace());
  Value *Addr = B.CreatePtrToInt(Ptr, B.getIntNTy(PtrBits));
  Value *Slot = B.CreateAnd(B.CreateLShr(Addr, 4), NumLocks - 1);
  Value *Lock = B.CreateInBoundsGEP(Locks->getValueType(), Locks,
                                    {B.getInt64(0), B.CreateZExtOrTrunc(Slot, B.getInt64Ty())});

  B.SetInsertPoint(Head);
  Value *Acq = B.CreateAtomicCmpXchg(Lock, B.getInt32(0), B.getInt32(1), MaybeAlign(4),
                                     AtomicOrdering::Monotonic,
                                     AtomicOrdering::Monotonic);
  B.CreateCondBr(B.CreateExtractValue(Acq, 1), Crit, Latch);

  B.SetInsertPoint(Crit);
  // Lock semantics: acquire after taking the lock, release before freeing it.
  B.CreateFence(AtomicOrdering::Acquire);
  auto Word = [&](unsigned K) {
    return B.CreateConstGEP1_32(B.getInt8Ty(), Ptr, 4 * K);
  };
  Value *Loaded = ConstantInt::get(B.getIntNTy(Bits), 0);
  for (unsigned K = 0; K < Words; ++K) {
    Value *W = B.CreateAtomicRMW(AtomicRMWInst::Or, Word(K), B.getInt32(0),
                                 MaybeAlign(4), AtomicOrdering::Monotonic);
    W = B.CreateShl(B.CreateZExt(W, B.getIntNTy(Bits)), 32 * K);
    Loaded = B.CreateOr(Loaded, W);
  }
  Value *Old = fromInt(B, Loaded, ValTy);
  Value *Result = Old;
  Value *New = nullptr;     // the value to store
  Value *DoStore = nullptr; // whether to store it
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    New = SI->getValueOperand();
  } else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
    New = buildAtomicRMWValue(RMW->getOperation(), B, Old, RMW->getValOperand());
  } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
    Value *Eq = B.CreateICmpEQ(Loaded, toInt(B, CX->getCompareOperand(), Bits));
    New = CX->getNewValOperand();
    DoStore = Eq;
    Value *Pair = B.CreateInsertValue(PoisonValue::get(CX->getType()), Old, 0);
    Result = B.CreateInsertValue(Pair, Eq, 1);
  }
  if (New) {
    Value *NewInt = toInt(B, New, Bits);
    BasicBlock *StoreBB = nullptr, *After = nullptr;
    if (DoStore) {
      StoreBB = BasicBlock::Create(Ctx, "wide.atomic.store", F, Latch);
      After = BasicBlock::Create(Ctx, "wide.atomic.unlock", F, Latch);
      B.CreateCondBr(DoStore, StoreBB, After);
      B.SetInsertPoint(StoreBB);
    }
    for (unsigned K = 0; K < Words; ++K) {
      Value *W = B.CreateTrunc(B.CreateLShr(NewInt, 32 * K), I32);
      B.CreateAtomicRMW(AtomicRMWInst::Xchg, Word(K), W, MaybeAlign(4),
                        AtomicOrdering::Monotonic);
    }
    if (DoStore) {
      B.CreateBr(After);
      B.SetInsertPoint(After);
    }
  }
  B.CreateFence(AtomicOrdering::Release);
  B.CreateAtomicRMW(AtomicRMWInst::Xchg, Lock, B.getInt32(0), MaybeAlign(4),
                    AtomicOrdering::Monotonic);
  BasicBlock *CritEnd = B.GetInsertBlock();
  B.CreateBr(Latch);

  Type *ResTy = isa<AtomicCmpXchgInst>(I) ? I.getType() : ValTy;
  B.SetInsertPoint(Latch);
  PHINode *Got = B.CreatePHI(B.getInt1Ty(), 2);
  Got->addIncoming(B.getTrue(), CritEnd);
  Got->addIncoming(B.getFalse(), Head);
  PHINode *Res = B.CreatePHI(ResTy, 2);
  Res->addIncoming(Result, CritEnd);
  // The result of a thread that has not done its work yet is not used.
  Res->addIncoming(PoisonValue::get(ResTy), Head);
  B.CreateCondBr(Got, Exit, Head);

  if (!I.getType()->isVoidTy()) {
    B.SetInsertPoint(&I);
    PHINode *Out = B.CreatePHI(ResTy, 1);
    Out->addIncoming(Res, Latch);
    I.replaceAllUsesWith(Out);
  }
  I.eraseFromParent();
}

bool AGXExpandWideAtomics::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const DataLayout &DL = F.getDataLayout();
  const AGXTargetLowering *TLI =
      TM ? TM->getSubtargetImpl(F)->getTargetLowering() : nullptr;
  SmallVector<std::pair<Instruction *, unsigned>, 4> Wide;
  for (Instruction &I : instructions(F)) {
    // Wide operations the chip has an instruction for.
    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I);
        RMW && TLI &&
        TLI->shouldExpandAtomicRMWInIR(RMW) ==
            TargetLowering::AtomicExpansionKind::None)
      continue;
    if (unsigned Bits = wideAtomicSize(I, DL))
      Wide.push_back({&I, Bits});
  }
  if (Wide.empty())
    return false;
  GlobalVariable *Locks = getLocks(*F.getParent());
  for (auto [I, Bits] : Wide)
    expand(*I, Bits, Locks);
  return true;
}
