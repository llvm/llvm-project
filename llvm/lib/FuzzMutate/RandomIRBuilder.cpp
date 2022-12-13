//===-- RandomIRBuilder.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/FuzzMutate/Random.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;
using namespace fuzzerop;

Value *RandomIRBuilder::findOrCreateSource(BasicBlock &BB,
                                           ArrayRef<Instruction *> Insts) {
  return findOrCreateSource(BB, Insts, {}, anyType());
}

Value *RandomIRBuilder::findOrCreateSource(BasicBlock &BB,
                                           ArrayRef<Instruction *> Insts,
                                           ArrayRef<Value *> Srcs,
                                           SourcePred Pred,
                                           bool allowConstant) {
  auto MatchesPred = [&Srcs, &Pred](Instruction *Inst) {
    return Pred.matches(Srcs, Inst);
  };
  auto RS = makeSampler(Rand, make_filter_range(Insts, MatchesPred));
  // Also consider choosing no source, meaning we want a new one.
  RS.sample(nullptr, /*Weight=*/1);
  if (Instruction *Src = RS.getSelection())
    return Src;
  return newSource(BB, Insts, Srcs, Pred, allowConstant);
}

Value *RandomIRBuilder::newSource(BasicBlock &BB, ArrayRef<Instruction *> Insts,
                                  ArrayRef<Value *> Srcs, SourcePred Pred,
                                  bool allowConstant) {
  // Generate some constants to choose from.
  auto RS = makeSampler<Value *>(Rand);
  RS.sample(Pred.generate(Srcs, KnownTypes));

  // If we can find a pointer to load from, use it half the time.
  Value *Ptr = findPointer(BB, Insts, Srcs, Pred);
  if (Ptr) {
    // Create load from the chosen pointer
    auto IP = BB.getFirstInsertionPt();
    if (auto *I = dyn_cast<Instruction>(Ptr)) {
      IP = ++I->getIterator();
      assert(IP != BB.end() && "guaranteed by the findPointer");
    }
    // For opaque pointers, pick the type independently.
    Type *AccessTy = Ptr->getType()->isOpaquePointerTy()
                         ? RS.getSelection()->getType()
                         : Ptr->getType()->getNonOpaquePointerElementType();
    auto *NewLoad = new LoadInst(AccessTy, Ptr, "L", &*IP);

    // Only sample this load if it really matches the descriptor
    if (Pred.matches(Srcs, NewLoad))
      RS.sample(NewLoad, RS.totalWeight());
    else
      NewLoad->eraseFromParent();
  }

  Value *newSrc = RS.getSelection();
  // Generate a stack alloca and store the constant to it if constant is not
  // allowed, our hope is that later mutations can generate some values and
  // store to this placeholder.
  if (!allowConstant && isa<Constant>(newSrc)) {
    Type *Ty = newSrc->getType();
    Function *F = BB.getParent();
    BasicBlock *EntryBB = &F->getEntryBlock();
    /// TODO: For all Allocas, maybe allocate an array.
    DataLayout DL(BB.getParent()->getParent());
    AllocaInst *Alloca = new AllocaInst(Ty, DL.getProgramAddressSpace(), "A",
                                        EntryBB->getTerminator());
    new StoreInst(newSrc, Alloca, EntryBB->getTerminator());
    if (BB.getTerminator()) {
      newSrc = new LoadInst(Ty, Alloca, /*ArrLen,*/ "L", BB.getTerminator());
    } else {
      newSrc = new LoadInst(Ty, Alloca, /*ArrLen,*/ "L", &BB);
    }
  }
  return newSrc;
}

static bool isCompatibleReplacement(const Instruction *I, const Use &Operand,
                                    const Value *Replacement) {
  unsigned int OperandNo = Operand.getOperandNo();
  if (Operand->getType() != Replacement->getType())
    return false;
  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:
  case Instruction::ExtractElement:
  case Instruction::ExtractValue:
    // TODO: We could potentially validate these, but for now just leave indices
    // alone.
    if (OperandNo >= 1)
      return false;
    break;
  case Instruction::InsertValue:
  case Instruction::InsertElement:
  case Instruction::ShuffleVector:
    if (OperandNo >= 2)
      return false;
    break;
  // For Br/Switch, we only try to modify the 1st Operand (condition).
  // Modify other operands, like switch case may accidently change case from
  // ConstantInt to a register, which is illegal.
  case Instruction::Switch:
  case Instruction::Br:
    if (OperandNo >= 1)
      return false;
    break;
  default:
    break;
  }
  return true;
}

void RandomIRBuilder::connectToSink(BasicBlock &BB,
                                    ArrayRef<Instruction *> Insts, Value *V) {
  auto RS = makeSampler<Use *>(Rand);
  for (auto &I : Insts) {
    if (isa<IntrinsicInst>(I))
      // TODO: Replacing operands of intrinsics would be interesting, but
      // there's no easy way to verify that a given replacement is valid given
      // that intrinsics can impose arbitrary constraints.
      continue;
    for (Use &U : I->operands())
      if (isCompatibleReplacement(I, U, V))
        RS.sample(&U, 1);
  }
  // Also consider choosing no sink, meaning we want a new one.
  RS.sample(nullptr, /*Weight=*/1);

  if (Use *Sink = RS.getSelection()) {
    User *U = Sink->getUser();
    unsigned OpNo = Sink->getOperandNo();
    U->setOperand(OpNo, V);
    return;
  }
  newSink(BB, Insts, V);
}

void RandomIRBuilder::newSink(BasicBlock &BB, ArrayRef<Instruction *> Insts,
                              Value *V) {
  Value *Ptr = findPointer(BB, Insts, {V}, matchFirstType());
  if (!Ptr) {
    if (uniform(Rand, 0, 1))
      Ptr = new AllocaInst(V->getType(), 0, "A", &*BB.getFirstInsertionPt());
    else
      Ptr = UndefValue::get(PointerType::get(V->getType(), 0));
  }

  new StoreInst(V, Ptr, Insts.back());
}

Value *RandomIRBuilder::findPointer(BasicBlock &BB,
                                    ArrayRef<Instruction *> Insts,
                                    ArrayRef<Value *> Srcs, SourcePred Pred) {
  auto IsMatchingPtr = [&Srcs, &Pred](Instruction *Inst) {
    // Invoke instructions sometimes produce valid pointers but currently
    // we can't insert loads or stores from them
    if (Inst->isTerminator())
      return false;

    if (auto *PtrTy = dyn_cast<PointerType>(Inst->getType())) {
      if (PtrTy->isOpaque())
        return true;

      // We can never generate loads from non first class or non sized types
      Type *ElemTy = PtrTy->getNonOpaquePointerElementType();
      if (!ElemTy->isSized() || !ElemTy->isFirstClassType())
        return false;

      // TODO: Check if this is horribly expensive.
      return Pred.matches(Srcs, UndefValue::get(ElemTy));
    }
    return false;
  };
  if (auto RS = makeSampler(Rand, make_filter_range(Insts, IsMatchingPtr)))
    return RS.getSelection();
  return nullptr;
}

Type *RandomIRBuilder::randomType() {
  uint64_t TyIdx = uniform<uint64_t>(Rand, 0, KnownTypes.size() - 1);
  return KnownTypes[TyIdx];
}
