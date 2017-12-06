//===- CallPromotionUtils.cpp - Utilities for call promotion ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities useful for promoting indirect call sites to
// direct call sites.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CallPromotionUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "call-promotion-utils"

/// Fix-up phi nodes in an invoke instruction's normal destination.
///
/// After versioning an invoke instruction, values coming from the original
/// block will now either be coming from the original block or the "else" block.
static void fixupPHINodeForNormalDest(InvokeInst *Invoke, BasicBlock *OrigBlock,
                                      BasicBlock *ElseBlock,
                                      Instruction *NewInst) {
  for (auto &I : *Invoke->getNormalDest()) {
    auto *Phi = dyn_cast<PHINode>(&I);
    if (!Phi)
      break;
    int Idx = Phi->getBasicBlockIndex(OrigBlock);
    if (Idx == -1)
      continue;
    Value *V = Phi->getIncomingValue(Idx);
    if (dyn_cast<Instruction>(V) == Invoke) {
      Phi->setIncomingBlock(Idx, ElseBlock);
      Phi->addIncoming(NewInst, OrigBlock);
      continue;
    }
    Phi->addIncoming(V, ElseBlock);
  }
}

/// Fix-up phi nodes in an invoke instruction's unwind destination.
///
/// After versioning an invoke instruction, values coming from the original
/// block will now be coming from either the "then" block or the "else" block.
static void fixupPHINodeForUnwindDest(InvokeInst *Invoke, BasicBlock *OrigBlock,
                                      BasicBlock *ThenBlock,
                                      BasicBlock *ElseBlock) {
  for (auto &I : *Invoke->getUnwindDest()) {
    auto *Phi = dyn_cast<PHINode>(&I);
    if (!Phi)
      break;
    int Idx = Phi->getBasicBlockIndex(OrigBlock);
    if (Idx == -1)
      continue;
    auto *V = Phi->getIncomingValue(Idx);
    Phi->setIncomingBlock(Idx, ThenBlock);
    Phi->addIncoming(V, ElseBlock);
  }
}

/// Get the phi node having the returned value of a call or invoke instruction
/// as it's operand.
static bool getRetPhiNode(Instruction *Inst, BasicBlock *Block) {
  BasicBlock *FromBlock = Inst->getParent();
  for (auto &I : *Block) {
    PHINode *PHI = dyn_cast<PHINode>(&I);
    if (!PHI)
      break;
    int Idx = PHI->getBasicBlockIndex(FromBlock);
    if (Idx == -1)
      continue;
    auto *V = PHI->getIncomingValue(Idx);
    if (V == Inst)
      return true;
  }
  return false;
}

/// Create a phi node for the returned value of a call or invoke instruction.
///
/// After versioning a call or invoke instruction that returns a value, we have
/// to merge the value of the original and new instructions. We do this by
/// creating a phi node and replacing uses of the original instruction with this
/// phi node.
static void createRetPHINode(Instruction *OrigInst, Instruction *NewInst) {

  if (OrigInst->getType()->isVoidTy() || OrigInst->use_empty())
    return;

  BasicBlock *RetValBB = NewInst->getParent();
  if (auto *Invoke = dyn_cast<InvokeInst>(NewInst))
    RetValBB = Invoke->getNormalDest();
  BasicBlock *PhiBB = RetValBB->getSingleSuccessor();

  if (getRetPhiNode(OrigInst, PhiBB))
    return;

  IRBuilder<> Builder(&PhiBB->front());
  PHINode *Phi = Builder.CreatePHI(OrigInst->getType(), 0);
  SmallVector<User *, 16> UsersToUpdate;
  for (User *U : OrigInst->users())
    UsersToUpdate.push_back(U);
  for (User *U : UsersToUpdate)
    U->replaceUsesOfWith(OrigInst, Phi);
  Phi->addIncoming(OrigInst, OrigInst->getParent());
  Phi->addIncoming(NewInst, RetValBB);
}

/// Cast a call or invoke instruction to the given type.
///
/// When promoting a call site, the return type of the call site might not match
/// that of the callee. If this is the case, we have to cast the returned value
/// to the correct type. The location of the cast depends on if we have a call
/// or invoke instruction.
Instruction *createRetBitCast(CallSite CS, Type *RetTy) {

  // Save the users of the calling instruction. These uses will be changed to
  // use the bitcast after we create it.
  SmallVector<User *, 16> UsersToUpdate;
  for (User *U : CS.getInstruction()->users())
    UsersToUpdate.push_back(U);

  // Determine an appropriate location to create the bitcast for the return
  // value. The location depends on if we have a call or invoke instruction.
  Instruction *InsertBefore = nullptr;
  if (auto *Invoke = dyn_cast<InvokeInst>(CS.getInstruction()))
    InsertBefore = &*Invoke->getNormalDest()->getFirstInsertionPt();
  else
    InsertBefore = &*std::next(CS.getInstruction()->getIterator());

  // Bitcast the return value to the correct type.
  auto *Cast = CastInst::Create(Instruction::BitCast, CS.getInstruction(),
                                RetTy, "", InsertBefore);

  // Replace all the original uses of the calling instruction with the bitcast.
  for (User *U : UsersToUpdate)
    U->replaceUsesOfWith(CS.getInstruction(), Cast);

  return Cast;
}

/// Predicate and clone the given call site.
///
/// This function creates an if-then-else structure at the location of the call
/// site. The "if" condition compares the call site's called value to the given
/// callee. The original call site is moved into the "else" block, and a clone
/// of the call site is placed in the "then" block. The cloned instruction is
/// returned.
static Instruction *versionCallSite(CallSite CS, Value *Callee,
                                    MDNode *BranchWeights,
                                    BasicBlock *&ThenBlock,
                                    BasicBlock *&ElseBlock,
                                    BasicBlock *&MergeBlock) {

  IRBuilder<> Builder(CS.getInstruction());
  Instruction *OrigInst = CS.getInstruction();

  // Create the compare. The called value and callee must have the same type to
  // be compared.
  auto *LHS =
      Builder.CreateBitCast(CS.getCalledValue(), Builder.getInt8PtrTy());
  auto *RHS = Builder.CreateBitCast(Callee, Builder.getInt8PtrTy());
  auto *Cond = Builder.CreateICmpEQ(LHS, RHS);

  // Create an if-then-else structure. The original instruction is moved into
  // the "else" block, and a clone of the original instruction is placed in the
  // "then" block.
  TerminatorInst *ThenTerm = nullptr;
  TerminatorInst *ElseTerm = nullptr;
  SplitBlockAndInsertIfThenElse(Cond, CS.getInstruction(), &ThenTerm, &ElseTerm,
                                BranchWeights);
  ThenBlock = ThenTerm->getParent();
  ElseBlock = ElseTerm->getParent();
  MergeBlock = OrigInst->getParent();

  ThenBlock->setName("if.true.direct_targ");
  ElseBlock->setName("if.false.orig_indirect");
  MergeBlock->setName("if.end.icp");

  Instruction *NewInst = OrigInst->clone();
  OrigInst->moveBefore(ElseTerm);
  NewInst->insertBefore(ThenTerm);

  // If the original call site is an invoke instruction, we have extra work to
  // do since invoke instructions are terminating.
  if (auto *OrigInvoke = dyn_cast<InvokeInst>(OrigInst)) {
    auto *NewInvoke = cast<InvokeInst>(NewInst);

    // Invoke instructions are terminating, so we don't need the terminator
    // instructions that were just created.
    ThenTerm->eraseFromParent();
    ElseTerm->eraseFromParent();

    // Branch from the "merge" block to the original normal destination.
    Builder.SetInsertPoint(MergeBlock);
    Builder.CreateBr(OrigInvoke->getNormalDest());

    // Now set the normal destination of new the invoke instruction to be the
    // "merge" block.
    NewInvoke->setNormalDest(MergeBlock);
  }

  return NewInst;
}

bool llvm::isLegalToPromote(CallSite CS, Function *Callee,
                            const char **FailureReason) {
  assert(!CS.getCalledFunction() && "Only indirect call sites can be promoted");

  // Check the return type. The callee's return value type must be bitcast
  // compatible with the call site's type.
  Type *CallRetTy = CS.getInstruction()->getType();
  Type *FuncRetTy = Callee->getReturnType();
  if (CallRetTy != FuncRetTy)
    if (!CastInst::isBitCastable(FuncRetTy, CallRetTy)) {
      if (FailureReason)
        *FailureReason = "Return type mismatch";
      return false;
    }

  // The number of formal arguments of the callee.
  unsigned NumParams = Callee->getFunctionType()->getNumParams();

  // Check the number of arguments. The callee and call site must agree on the
  // number of arguments.
  if (CS.arg_size() != NumParams && !Callee->isVarArg()) {
    if (FailureReason)
      *FailureReason = "The number of arguments mismatch";
    return false;
  }

  // Check the argument types. The callee's formal argument types must be
  // bitcast compatible with the corresponding actual argument types of the call
  // site.
  for (unsigned I = 0; I < NumParams; ++I) {
    Type *FormalTy = Callee->getFunctionType()->getFunctionParamType(I);
    Type *ActualTy = CS.getArgument(I)->getType();
    if (FormalTy == ActualTy)
      continue;
    if (!CastInst::isBitCastable(ActualTy, FormalTy)) {
      if (FailureReason)
        *FailureReason = "Argument type mismatch";
      return false;
    }
  }

  return true;
}

static void promoteCall(CallSite CS, Function *Callee, Instruction *&Cast) {
  assert(!CS.getCalledFunction() && "Only indirect call sites can be promoted");

  // Set the called function of the call site to be the given callee.
  CS.setCalledFunction(Callee);

  // Since the call site will no longer be direct, we must clear metadata that
  // is only appropriate for indirect calls. This includes !prof and !callees
  // metadata.
  CS.getInstruction()->setMetadata(LLVMContext::MD_prof, nullptr);
  CS.getInstruction()->setMetadata(LLVMContext::MD_callees, nullptr);

  // If the function type of the call site matches that of the callee, no
  // additional work is required.
  if (CS.getFunctionType() == Callee->getFunctionType())
    return;

  // Save the return types of the call site and callee.
  Type *CallSiteRetTy = CS.getInstruction()->getType();
  Type *CalleeRetTy = Callee->getReturnType();

  // Change the function type of the call site the match that of the callee.
  CS.mutateFunctionType(Callee->getFunctionType());

  // Inspect the arguments of the call site. If an argument's type doesn't
  // match the corresponding formal argument's type in the callee, bitcast it
  // to the correct type.
  for (Use &U : CS.args()) {
    unsigned ArgNo = CS.getArgumentNo(&U);
    Type *FormalTy = Callee->getFunctionType()->getParamType(ArgNo);
    Type *ActualTy = U.get()->getType();
    if (FormalTy != ActualTy) {
      auto *Cast = CastInst::Create(Instruction::BitCast, U.get(), FormalTy, "",
                                    CS.getInstruction());
      CS.setArgument(ArgNo, Cast);
    }
  }

  // If the return type of the call site doesn't match that of the callee, cast
  // the returned value to the appropriate type.
  if (!CallSiteRetTy->isVoidTy() && CallSiteRetTy != CalleeRetTy)
    Cast = createRetBitCast(CS, CallSiteRetTy);
}

Instruction *llvm::promoteCallWithIfThenElse(CallSite CS, Function *Callee,
                                             MDNode *BranchWeights) {

  // Version the indirect call site. If the called value is equal to the given
  // callee, 'NewInst' will be executed, otherwise the original call site will
  // be executed.
  BasicBlock *ThenBlock, *ElseBlock, *MergeBlock;
  Instruction *NewInst = versionCallSite(CS, Callee, BranchWeights, ThenBlock,
                                         ElseBlock, MergeBlock);

  // Promote 'NewInst' so that it directly calls the desired function.
  Instruction *Cast = NewInst;
  promoteCall(CallSite(NewInst), Callee, Cast);

  // If the original call site is an invoke instruction, we have to fix-up phi
  // nodes in the invoke's normal and unwind destinations.
  if (auto *OrigInvoke = dyn_cast<InvokeInst>(CS.getInstruction())) {
    fixupPHINodeForNormalDest(OrigInvoke, MergeBlock, ElseBlock, Cast);
    fixupPHINodeForUnwindDest(OrigInvoke, MergeBlock, ThenBlock, ElseBlock);
  }

  // Create a phi node for the returned value of the call site.
  createRetPHINode(CS.getInstruction(), Cast ? Cast : NewInst);

  // Return the new direct call.
  return NewInst;
}

#undef DEBUG_TYPE
