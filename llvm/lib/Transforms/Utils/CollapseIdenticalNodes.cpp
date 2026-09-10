//===- CollapseIdenticalNodes.cpp
//----------------------------------------------------===//
//
//  Pass by Barkir
//  description: collapsing identical nodes
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CollapseIdenticalNodes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <stdexcept>
// #include "llvm/Support/CFG.h"

#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"

#define DEBUG_TYPE "collapse-nodes"

#ifdef DEBUG
#define ON_DEBUG(code) code
#else
#define ON_DEBUG(code)
#endif

using namespace llvm;

using valueHashTab = std::unordered_map<Value *, int32_t>;

// table for collecting icmp (needed to proof && links between conditions)
static std::vector<Instruction *> icmpTable;
using icmpIterator = std::vector<Instruction *>::iterator;

llvm::raw_fd_ostream &operator<<(llvm::raw_fd_ostream &os, valueHashTab &tab) {

  for (size_t i = 0; i < tab.size(); i++) {
    os << "═════";
  }
  os << "\n";

  for (const auto &pair : tab) {
    os << "║ ";
    if (pair.first) {
      pair.first->printAsOperand(os, false);
      os << "\t\t";
    } else {
      os << "null";
    }
    os << "║ "
       << "\t" << pair.second << "║\n";
  }
  for (size_t i = 0; i < tab.size(); i++) {
    os << "═════";
  }
  os << "\n";
  return os;
}

void findAllPaths(
    BasicBlock *curr,   // current basic block
    BasicBlock *target, // target basic block
    std::vector<BasicBlock *>
        &CurPath, // current path we chose to go from current -> ... -> target
    std::vector<std::vector<BasicBlock *>> &AllPaths, // vector of all paths
    std::set<BasicBlock *>
        &Visited) { // needed to prevent cycling if we have cycle in a block
                    // (e.g. current -> current -> ...)
  if (curr == nullptr) {
    return;
  }
  CurPath.push_back(curr);
  Visited.insert(curr);

  if (curr == target) {
    for (auto path : CurPath) {
      errs() << path->getName() << "->";
    }
    errs() << "\n";
    AllPaths.push_back(CurPath);
    return;
  }

  for (BasicBlock *succ : successors(curr)) {
    if (Visited.find(succ) == Visited.end()) {
      errs() << succ->getName() << " BB IS IN PATH"
             << "\n";
      findAllPaths(succ, target, CurPath, AllPaths, Visited);
    }
  }

  CurPath.pop_back();
  Visited.erase(curr);
}

int CountValueRecursively(Value *val, valueHashTab &tab) {
  errs() << UCYN "CountValueRecursively of " << *val << RESET << "\n";
  int retVal = 0;
  auto value = tab.find(val);
  if (value != tab.end()) {
    errs() << "Found value in hashtable " << BYEL << value->second
           << RESET "\n";
    return value->second;
  }
  if (auto intVal = dyn_cast<ConstantInt>(val)) {
    errs() << "The value from instruction is " << BYEL << *intVal << RESET "\n";
    return intVal->getSExtValue();
  }

  auto *instr = dyn_cast<BinaryOperator>(val);
  if (instr) {
    switch (instr->getOpcode()) {
    case Instruction::Add:
      errs() << YEL "ADD " << *instr << RESET "\n";
      retVal = CountValueRecursively(instr->getOperand(0), tab) +
               CountValueRecursively(instr->getOperand(1), tab);
      break;

    case Instruction::Sub:
      errs() << YEL "SUB " << *instr << RESET "\n";
      retVal = CountValueRecursively(instr->getOperand(0), tab) -
               CountValueRecursively(instr->getOperand(1), tab);
      break;

    case Instruction::Mul:
      errs() << YEL "MUL " << *instr << RESET "\n";
      retVal = CountValueRecursively(instr->getOperand(0), tab) *
               CountValueRecursively(instr->getOperand(1), tab);
      break;

    case Instruction::SDiv:
      errs() << YEL "DIV " << *instr << RESET "\n";
      auto right = CountValueRecursively(instr->getOperand(1), tab);
      if (right == 0) // FIXME - bad error handling
        return retVal;

      retVal = CountValueRecursively(instr->getOperand(0), tab) /
               CountValueRecursively(instr->getOperand(1), tab);

      break;
    }
  }
  return retVal;
}

int GetConstantFromPhiNode(Instruction *phi) {
  errs() << UCYN "GetConstantFromPhiNode" RESET << "\n";
  auto *phiNode = dyn_cast<PHINode>(phi);
  if (phiNode) {
    int numValues = phiNode->getNumIncomingValues();
    for (int i = 0; i < numValues; i++) {
      if (auto *constantVal =
              dyn_cast<ConstantInt>(phiNode->getIncomingValue(i))) {
        return constantVal->getSExtValue();
      }
    }
  }
}

Instruction *PhiNodeWithConstant(BasicBlock &BB) {
  errs() << UCYN "PhiNodeWithConstant" RESET << "\n";
  for (auto &I : BB) {
    errs() << BYEL "Got instruction " << I << RESET "\n";
    if (auto *phiInstr = dyn_cast<PHINode>(&I)) {
      errs() << "Found phi node " << *phiInstr << "\n";
      int numValues = phiInstr->getNumIncomingValues();
      for (int i = 0; i < numValues; i++) {
        if (auto *constantVal =
                dyn_cast<ConstantInt>(phiInstr->getIncomingValue(i))) {
          errs() << "Phi node has a constant. Returning it as an Instruction*"
                 << "\n";
          errs() << UCYN "END OF FUNC" RESET << "\n";
          return &I;
        }
      }
    }
  }
  errs() << UCYN "END OF FUNC" RESET << "\n";
  return nullptr;
}

Value *getPznOrConstant(Value *val, valueHashTab &tab) {
  errs() << UCYN "getPznOrConstant" RESET << "\n";
  LLVMContext &ctx = val->getContext();

  // finding values in hash table (depth = 2);
  auto hfind_val = tab.find(val);

  // explained on this example:
  // %val = {instr} i32 %val1, %val2;
  // {instr} is '+', '-', '*', '/'
  if (hfind_val != tab.end()) { // a case where %val is already in a hashtable
                                // (return: tab[%val])
    errs() << BYEL "Value " RESET << hfind_val->first
           << BYEL " is in hash table - " << BWHT << hfind_val->second
           << RESET "\n";
    ConstantInt *ret =
        ConstantInt::get(Type::getInt32Ty(ctx), hfind_val->second, true);
    return ret;
  }
  if (auto *instrVal = dyn_cast<Instruction>(val)) {
    auto hfind_fop = tab.find(instrVal->getOperand(0));
    if (hfind_fop !=
        tab.end()) { // a case where %val1 is in a hashtable and %val2 is a
                     // constant (return: tab[%val1] {instr} %val2)
      errs() << BYEL "Value " RESET << *hfind_fop->first
             << BYEL " is in hash table - " << BWHT << hfind_fop->second
             << RESET "\n";
      auto *foundInstr = dyn_cast<BinaryOperator>(instrVal);
      if (foundInstr) {
        auto *secondOperandConst =
            dyn_cast<ConstantInt>(foundInstr->getOperand(1));

        if (secondOperandConst) {
          errs() << BYEL "second operand is a ConstantInt : first operand is "
                         "an Instruction"
                 << "RESET"
                 << "\n";
          int32_t firstConst = hfind_fop->second;
          auto secondConst = secondOperandConst->getSExtValue();

          int32_t retInt = 0;
          ConstantInt *ret = nullptr;

          switch (foundInstr->getOpcode()) {
          case Instruction::Add:
            retInt = firstConst + secondConst;
            errs() << GRNHB "ADD = " << retInt << "\n";
            ret = ConstantInt::get(Type::getInt32Ty(ctx), retInt, true);
            errs() << UCYN "END OF FUNC" RESET << "\n";
            return ret;

          case Instruction::Sub:
            retInt = firstConst - secondConst;
            errs() << GRNHB "SUB = " << retInt << "\n";
            ret = ConstantInt::get(Type::getInt32Ty(ctx), retInt, true);
            errs() << UCYN "END OF FUNC" RESET << "\n";
            return ret;

          case Instruction::Mul:
            retInt = firstConst * secondConst;
            errs() << GRNHB "MUL = " << retInt << "\n";
            ret = ConstantInt::get(Type::getInt32Ty(ctx), retInt, true);
            errs() << UCYN "END OF FUNC" RESET << "\n";
            return ret;

          case Instruction::SDiv:
            retInt = firstConst / secondConst;
            errs() << GRNHB "SDiv = " << retInt << "\n";
            ret = ConstantInt::get(Type::getInt32Ty(ctx), retInt, true);
            errs() << UCYN "END OF FUNC" RESET << "\n";
            return ret;
          }
        }
      }
    }
  }
  Value *pzn = PoisonValue::get(Type::getInt32Ty(ctx));
  errs() << UCYN "END OF FUNC" RESET << "\n";
  return pzn;
}

valueHashTab collectICmp(BasicBlock &BB, DominatorTree &DT) {
  errs() << UCYN "collectICmp" RESET << "\n";
  valueHashTab instrMap;
  const DomTreeNode *Node = DT.getNode(&BB);
  DomTreeNode *IDom = Node->getIDom();
  BasicBlock *ImmBB = IDom->getBlock();

  std::vector<BasicBlock *> curPath;
  std::vector<std::vector<BasicBlock *>> allPaths;
  std::set<BasicBlock *> visited;

  findAllPaths(ImmBB, &BB, curPath, allPaths, visited);
  errs() << "FOUND ALL PATHS!!!"
         << "\n";
  for (auto &path : allPaths) {
    for (auto pathBB : path) {
      errs() << pathBB->getName() << "IN PATH CYCLE"
             << "\n";
      for (auto &I : *pathBB) {
        if (I.getOpcode() == Instruction::ICmp) {
          errs() << BYEL "Found icmp! " << I << RESET "\n";
          auto *castI = dyn_cast<ICmpInst>(&I);
          if (castI->getPredicate() == CmpInst::ICMP_EQ) {
            errs() << "That is icmp eq!"
                   << "\n";
            auto *firstOperand = I.getOperand(0);
            auto *secondOperand = I.getOperand(1);
            auto *secondFromTab = getPznOrConstant(secondOperand, instrMap);
            if (auto *secondOperandConst =
                    dyn_cast<ConstantInt>(secondOperand)) {
              errs() << BYEL << "second op is num! ";
              const int32_t sextVal = secondOperandConst->getSExtValue();
              errs() << CYNHB "sextVal is " << sextVal << RESET "\n";
              instrMap[firstOperand] = sextVal;
              icmpTable.push_back(castI);
            } else if (!isa<PoisonValue>(secondFromTab)) {
              auto *secondFromTabConst = dyn_cast<ConstantInt>(secondFromTab);
              errs() << BYEL << "secondOperand is a complexed value!" << RESET
                     << "\n";
              const int32_t sextVal = secondFromTabConst->getSExtValue();
              errs() << CYNHB "sextVal is " << sextVal << RESET "\n";
              instrMap[firstOperand] = sextVal;
              icmpTable.push_back(castI);
            } else if (auto *firstOperandConst =
                           dyn_cast<ConstantInt>(firstOperand)) {
              errs() << BYEL << "first op is num! ";
              const int32_t sextVal = firstOperandConst->getSExtValue();
              errs() << CYNHB "sextVal is " << sextVal << RESET "\n";
              instrMap[secondOperand] = sextVal;
              icmpTable.push_back(castI);
            }
          }
        }
      }
    }
  }
  errs() << instrMap;
  errs() << UCYN "END OF FUNC" RESET << "\n";
  return instrMap;
}

bool checkLinkThruTerminator(icmpIterator pred, icmpIterator post) {
  errs() << UCYN "checkLinkThruTerminator" RESET << "\n";
  auto predObj = *pred;
  if (!predObj)
    return false;

  auto predParent = predObj->getParent();
  if (!predParent)
    return false;

  auto predBBTerminator = predParent->getTerminator();

  auto postObj = *post;
  if (!postObj)
    return false;

  auto postBB = postObj->getParent();
  if (!postBB)
    return false;

  errs() << "Terminator of predBB: " << *predBBTerminator << "\n";
  errs() << "Comparing it's operand with postBB"
         << "\n";
  errs() << *(predBBTerminator->getOperand(2)) << *postBB << "\n";
  if (predBBTerminator->getOperand(2) == postBB) {
    errs() << CYNHB "EQUAL!" << RESET "\n";
    errs() << UCYN "END OF FUNC" RESET << "\n";
    return true;
  }
  errs() << UCYN "END OF FUNC" RESET << "\n";
  return false;
}

bool checkLinkThruAnd(icmpIterator first, icmpIterator second) {
  errs() << UCYN "checkLinkThruAnd" RESET << "\n";

  auto firstObj = *first;
  auto secondObj = *second;

  if (!firstObj || !secondObj) {
    errs() << "One of the instructions is null\n";
    return false;
  }

  auto firstBB = firstObj->getParent();
  auto secondBB = secondObj->getParent();

  if (!firstBB || !secondBB) {
    errs() << "One of the instructions has no parent block\n";
    return false;
  }

  if (firstBB != secondBB) {
    errs() << "Instructions are in different basic blocks\n";
    return false;
  }

  errs() << "Searching for AND instruction in BB: " << firstBB->getName()
         << "\n";

  for (auto &inst : *firstBB) {
    if (inst.getOpcode() == Instruction::And) {
      Value *op0 = inst.getOperand(0);
      Value *op1 = inst.getOperand(1);

      bool match1 = (op0 == firstObj && op1 == secondObj);
      bool match2 = (op0 == secondObj && op1 == firstObj);

      if (match1 || match2) {
        errs() << CYNHB "FOUND AND CONNECTION!" << RESET << "\n";
        errs() << "AND Instruction: " << inst << "\n";
        errs() << UCYN "END OF FUNC" RESET << "\n";
        return true;
      }
    }
  }

  errs() << "No AND instruction connects these two values\n";
  errs() << UCYN "END OF FUNC" RESET << "\n";
  return false;
}

bool checkLinkThruSelect(icmpIterator pred, icmpIterator post) {
  errs() << UCYN "checkLinkThruSelect" RESET << "\n";
  if ((*pred)->getParent() ==
      (*post)->getParent()) { // instructions are in the same basic block
    for (auto user : (*pred)->users()) {
      errs() << "Got User " << *user << "\n";
      if (auto userToInst = dyn_cast<Instruction>(user)) {
        if (userToInst->getOpcode() == Instruction::Select) {
          errs() << "user is select!"
                 << "\n";
          if (userToInst->getOperand(0) == *pred &&
              userToInst->getOperand(1) == *post) {
            *post = userToInst; // updating because select result will be used
                                // in future BB's, not the old post
            return true;
          }
        }
      }
    }
  }
  errs() << UCYN "END OF FUNC" RESET << "\n";
  return false;
}

bool proofConjuctionLinks() {
  errs() << UCYN "proofConjuctionLinks" RESET << "\n";
  for (auto it = icmpTable.begin(), end = icmpTable.end() - 1; it != end;
       ++it) {
    errs() << **it << **(it + 1) << "\n";
    if (!checkLinkThruTerminator(it, it + 1) &&
        !checkLinkThruSelect(it, it + 1) && !checkLinkThruAnd(it, it + 1)) {
      return false;
    }
  }
  return true;
  errs() << UCYN "END OF FUNC" RESET << "\n";
}

/*
incoming block:
for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %x.0, %y.0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !6

 */
void DeprecateConstantFromPhiNode(PHINode *phi, BasicBlock *target) {
  errs() << UCYN "DeprecateConstantFromPhiNode" RESET << "\n";
  errs() << *phi << "\n";
  const int32_t numValues = phi->getNumIncomingValues();
  for (int i = 0; i < numValues; ++i) {
    if (auto *incomingValue = dyn_cast<ConstantInt>(phi->getIncomingValue(i))) {
      BasicBlock *blockToDelete = phi->getIncomingBlock(i);
      for (BasicBlock *pred : predecessors(blockToDelete)) {
        for (PHINode &targetPhi : target->phis()) {

          if (unsigned idx = targetPhi.getBasicBlockIndex(blockToDelete)) {
            Value *valFromDeleted = targetPhi.getIncomingValue(idx);

            targetPhi.addIncoming(valFromDeleted, pred);
          }
        }
      }

      for (pred_iterator PI = pred_begin(blockToDelete);
           PI != pred_end(blockToDelete); ++PI) {
        BasicBlock *pred = *PI;
        pred->getTerminator()->replaceSuccessorWith(blockToDelete, target);
      }

      for (PHINode &targetPhi : target->phis()) {
        if (int idx = targetPhi.getBasicBlockIndex(blockToDelete); idx != -1) {
          targetPhi.removeIncomingValue(idx);
        }
      }

      DeleteDeadBlock(blockToDelete);

      break;
    }
  }
  errs() << UCYN "END OF FUNC" RESET << "\n";
}

PreservedAnalyses CollapseIdenticalNodesPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {

  errs() << "Starting CollapseIdenticalNodesPass..."
         << "\n";

  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  for (auto &BB : F) {
    if (auto *phiNode = PhiNodeWithConstant(BB)) {
      valueHashTab tab = collectICmp(BB, DT);
      errs() << "collected icmp and got hashtab!"
             << "\n";
      bool links = proofConjuctionLinks();
      if (links) {
        auto *phi = dyn_cast<PHINode>(phiNode);
        auto *phiToReplace = phi->getIncomingValue(1);
        auto value = CountValueRecursively(phiToReplace, tab);
        errs() << MAGB "Total Value is " << value << RESET "\n";
        if (value == GetConstantFromPhiNode(phiNode)) {
          if (auto *phiToReplaceInstr = dyn_cast<Instruction>(phiToReplace)) {
            BasicBlock *incomingBlock = phi->getIncomingBlock(1);
            errs() << "incoming block: " << *incomingBlock << "\n";
            DeprecateConstantFromPhiNode(phi, incomingBlock);
            errs() << F << "\n";
          }
          errs() << "VALUES ARE EQUAL"
                 << "\n";
          return PreservedAnalyses::all();
        }
      }
    }
  }
  return PreservedAnalyses::none();
}
