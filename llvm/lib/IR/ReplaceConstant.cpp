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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {

void convertConstantExprsToInstructions(Instruction *I, ConstantExpr *CE,
                                        SmallPtrSetImpl<Instruction *> *Insts) {
  // Collect all reachable paths to CE from constant exprssion operands of I.
  std::map<Use *, std::vector<std::vector<ConstantExpr *>>> CEPaths;
  collectConstantExprPaths(I, CE, CEPaths);

  // Convert all constant expressions to instructions which are collected at
  // CEPaths.
  convertConstantExprsToInstructions(I, CEPaths, Insts);
}

void convertConstantExprsToInstructions(
    Instruction *I,
    std::map<Use *, std::vector<std::vector<ConstantExpr *>>> &CEPaths,
    SmallPtrSetImpl<Instruction *> *Insts) {
  ValueMap<ConstantExpr *, Instruction *> Visited;

  for (Use &U : I->operands()) {
    // The operand U is either not a constant expression operand or the
    // constant expression paths do not belong to U, ignore U.
    if (!CEPaths.count(&U))
      continue;

    // If the instruction I is a PHI instruction, then fix the instruction
    // insertion point to the entry of the incoming basic block for operand U.
    auto *BI = I;
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      BasicBlock *BB = Phi->getIncomingBlock(U);
      BI = &(*(BB->getFirstInsertionPt()));
    }

    // Go through all the paths associated with operand U, and convert all the
    // constant expressions along all the paths to corresponding instructions.
    auto *II = I;
    auto &Paths = CEPaths[&U];
    for (auto &Path : Paths) {
      for (auto *CE : Path) {
        // Instruction which is equivalent to CE.
        Instruction *NI = nullptr;

        if (!Visited.count(CE)) {
          // CE is encountered first time, convert it into a corresponding
          // instruction NI, and appropriately insert NI before the parent
          // instruction.
          NI = CE->getAsInstruction(BI);

          // Mark CE as visited by mapping CE to NI.
          Visited[CE] = NI;

          // If required collect NI.
          if (Insts)
            Insts->insert(NI);
        } else {
          // We had already encountered CE, the correponding instruction already
          // exist, use it to replace CE.
          NI = Visited[CE];
        }

        assert(NI && "Expected an instruction corresponding to constant "
                     "expression.");

        // Replace all uses of constant expression CE by the corresponding
        // instruction NI within the current parent instruction.
        II->replaceUsesOfWith(CE, NI);
        BI = II = NI;
      }
    }
  }

  // Remove all converted constant expressions which are dead by now.
  for (auto Item : Visited)
    Item.first->removeDeadConstantUsers();
}

void collectConstantExprPaths(
    Instruction *I, ConstantExpr *CE,
    std::map<Use *, std::vector<std::vector<ConstantExpr *>>> &CEPaths) {
  for (Use &U : I->operands()) {
    // If the operand U is not a constant expression operand, then ignore it.
    auto *CE2 = dyn_cast<ConstantExpr>(U.get());
    if (!CE2)
      continue;

    // Holds all reachable paths from CE2 to CE.
    std::vector<std::vector<ConstantExpr *>> Paths;

    // Collect all reachable paths from CE2 to CE.
    std::vector<ConstantExpr *> Path{CE2};
    std::vector<std::vector<ConstantExpr *>> Stack{Path};
    while (!Stack.empty()) {
      std::vector<ConstantExpr *> TPath = Stack.back();
      Stack.pop_back();
      auto *CE3 = TPath.back();

      if (CE3 == CE) {
        Paths.push_back(TPath);
        continue;
      }

      for (auto &UU : CE3->operands()) {
        if (auto *CE4 = dyn_cast<ConstantExpr>(UU.get())) {
          std::vector<ConstantExpr *> NPath(TPath.begin(), TPath.end());
          NPath.push_back(CE4);
          Stack.push_back(NPath);
        }
      }
    }

    // Associate all the collected paths with U, and save it.
    if (!Paths.empty())
      CEPaths[&U] = Paths;
  }
}

static bool isExpandableUser(User *U) {
  return isa<ConstantExpr>(U) || isa<ConstantAggregate>(U);
}

static Instruction *expandUser(Instruction *InsertPt, Constant *C) {
  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    return CE->getAsInstruction(InsertPt);
  } else if (isa<ConstantStruct>(C) || isa<ConstantArray>(C)) {
    Value *V = PoisonValue::get(C->getType());
    for (auto [Idx, Op] : enumerate(C->operands()))
      V = InsertValueInst::Create(V, Op, Idx, "", InsertPt);
    return cast<Instruction>(V);
  } else if (isa<ConstantVector>(C)) {
    Type *IdxTy = Type::getInt32Ty(C->getContext());
    Value *V = PoisonValue::get(C->getType());
    for (auto [Idx, Op] : enumerate(C->operands()))
      V = InsertElementInst::Create(V, Op, ConstantInt::get(IdxTy, Idx), "",
                                    InsertPt);
    return cast<Instruction>(V);
  } else {
    llvm_unreachable("Not an expandable user");
  }
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
          Instruction *NI = expandUser(BI, C);
          InstructionWorklist.insert(NI);
          U.set(NI);
        }
      }
    }
  }

  for (Constant *C : Consts)
    C->removeDeadConstantUsers();

  return Changed;
}

} // namespace llvm
