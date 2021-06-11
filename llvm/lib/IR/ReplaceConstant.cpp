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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/NoFolder.h"

namespace llvm {
// Replace a constant expression by instructions with equivalent operations at
// a specified location.
Instruction *createReplacementInstr(ConstantExpr *CE, Instruction *Instr) {
  auto *CEInstr = CE->getAsInstruction();
  CEInstr->insertBefore(Instr);
  return CEInstr;
}

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
  SmallPtrSet<ConstantExpr *, 8> Visited;
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

    // Go through the paths associated with operand U, and convert all the
    // constant expressions along all paths to corresponding instructions.
    auto *II = I;
    auto &Paths = CEPaths[&U];
    for (auto &Path : Paths) {
      for (auto *CE : Path) {
        if (!Visited.insert(CE).second)
          continue;
        auto *NI = CE->getAsInstruction();
        NI->insertBefore(BI);
        II->replaceUsesOfWith(CE, NI);
        CE->removeDeadConstantUsers();
        BI = II = NI;
        if (Insts)
          Insts->insert(NI);
      }
    }
  }
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

} // namespace llvm
