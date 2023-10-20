//===- ReplaceConstant.cpp - Replace LLVM constant expression--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility function for replacing LLVM constant
// expressions by instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ReplaceConstant.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

namespace llvm {

static bool isExpandableUser(User *U) {
  return isa<ConstantExpr>(U) || isa<ConstantAggregate>(U);
}

static SmallVector<Instruction *, 4> expandUser(Instruction *InsertPt,
                                                Constant *C) {
  SmallVector<Instruction *, 4> NewInsts;
  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    NewInsts.push_back(CE->getAsInstruction(InsertPt));
  } else if (isa<ConstantStruct>(C) || isa<ConstantArray>(C)) {
    Value *V = PoisonValue::get(C->getType());
    for (auto [Idx, Op] : enumerate(C->operands())) {
      V = InsertValueInst::Create(V, Op, Idx, "", InsertPt);
      NewInsts.push_back(cast<Instruction>(V));
    }
  } else if (isa<ConstantVector>(C)) {
    Type *IdxTy = Type::getInt32Ty(C->getContext());
    Value *V = PoisonValue::get(C->getType());
    for (auto [Idx, Op] : enumerate(C->operands())) {
      V = InsertElementInst::Create(V, Op, ConstantInt::get(IdxTy, Idx), "",
                                    InsertPt);
      NewInsts.push_back(cast<Instruction>(V));
    }
  } else {
    llvm_unreachable("Not an expandable user");
  }
  return NewInsts;
}

bool convertUsersOfConstantsToInstructions(ArrayRef<Constant *> Consts) {
  // Find all expandable direct users of Consts.
  SmallVector<Constant *> Stack;
  for (Constant *C : Consts)
    for (User *U : C->users())
      if (isExpandableUser(U))
        Stack.push_back(cast<Constant>(U));

  // Include transitive users.
  SetVector<Constant *> ExpandableUsers;
  while (!Stack.empty()) {
    Constant *C = Stack.pop_back_val();
    if (!ExpandableUsers.insert(C))
      continue;

    for (auto *Nested : C->users())
      if (isExpandableUser(Nested))
        Stack.push_back(cast<Constant>(Nested));
  }

  // Find all instructions that use any of the expandable users
  SetVector<Instruction *> InstructionWorklist;
  for (Constant *C : ExpandableUsers)
    for (User *U : C->users())
      if (auto *I = dyn_cast<Instruction>(U))
        InstructionWorklist.insert(I);

  // Replace those expandable operands with instructions
  bool Changed = false;
  while (!InstructionWorklist.empty()) {
    Instruction *I = InstructionWorklist.pop_back_val();
    DebugLoc Loc = I->getDebugLoc();
    for (Use &U : I->operands()) {
      auto *BI = I;
      if (auto *Phi = dyn_cast<PHINode>(I)) {
        BasicBlock *BB = Phi->getIncomingBlock(U);
        BasicBlock::iterator It = BB->getFirstInsertionPt();
        assert(It != BB->end() && "Unexpected empty basic block");
        BI = &*It;
      }

      if (auto *C = dyn_cast<Constant>(U.get())) {
        if (ExpandableUsers.contains(C)) {
          Changed = true;
          auto NewInsts = expandUser(BI, C);
          for (auto *NI : NewInsts)
            NI->setDebugLoc(Loc);
          InstructionWorklist.insert(NewInsts.begin(), NewInsts.end());
          U.set(NewInsts.back());
        }
      }
    }
  }

  for (Constant *C : Consts)
    C->removeDeadConstantUsers();

  return Changed;
}

} // namespace llvm
