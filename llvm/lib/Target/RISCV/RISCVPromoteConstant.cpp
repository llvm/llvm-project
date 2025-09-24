//==- RISCVPromoteConstant.cpp - Promote constant fp to global for RISC-V --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-promote-const"

STATISTIC(NumPromoted, "Number of promoted constants");
STATISTIC(NumPromotedUses, "Number of promoted constants uses");

namespace {

class RISCVPromoteConstant : public ModulePass {
public:
  static char ID;
  RISCVPromoteConstant() : ModulePass(ID) {}

  StringRef getPassName() const override { return "RISC-V Promote Constant"; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  /// Iterate over the functions and promote the double fp constants that
  /// would otherwise go into the constant pool to a constant array.
  bool runOnModule(Module &M) override {
    LLVM_DEBUG(dbgs() << getPassName() << '\n');
    // TargetMachine and Subtarget are needed to query isFPImmlegal. Get them
    // from TargetPassConfig.
    const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
    const TargetMachine &TM = TPC.getTM<TargetMachine>();
    if (skipModule(M))
      return false;
    bool Changed = false;
    for (auto &MF : M) {
      const RISCVSubtarget &ST = TM.getSubtarget<RISCVSubtarget>(MF);
      const RISCVTargetLowering *TLI = ST.getTargetLowering();
      Changed |= runOnFunction(MF, TLI);
    }
    return Changed;
  }

private:
  bool runOnFunction(Function &F, const RISCVTargetLowering *TLI);
};
} // end anonymous namespace

char RISCVPromoteConstant::ID = 0;

ModulePass *llvm::createRISCVPromoteConstantPass() {
  return new RISCVPromoteConstant();
}

bool RISCVPromoteConstant::runOnFunction(Function &F, const RISCVTargetLowering *TLI) {
  // Bail out and make no transformation if the target doesn't support
  // doubles, or if we're not targeting RV64 as we currently see some
  // regressions for those targets.
  if (!TLI->isTypeLegal(MVT::f64) || !TLI->isTypeLegal(MVT::i64))
    return false;

  // Collect all unique double constants used in the function, and track their
  // offset within the newly created global array. Also track uses that will
  // be replaced later.
  DenseMap<ConstantFP *, unsigned> ConstantMap;
  SmallVector<Constant *, 16> ConstantVector;
  DenseMap<ConstantFP *, SmallVector<Use *, 8>> UsesInFunc;

  for (Instruction &I : instructions(F)) {
    // PHI nodes are handled specially in a second loop below.
    if (isa<PHINode>(I))
      continue;
    for (Use &U : I.operands()) {
      if (auto *C = dyn_cast<ConstantFP>(U.get())) {
        if (C->getType()->isDoubleTy()) {
          if (TLI->isFPImmLegal(C->getValueAPF(), MVT::f64, /*ForCodeSize*/ false))
            continue;
          UsesInFunc[C].push_back(&U);
          if (ConstantMap.find(C) == ConstantMap.end()) {
            ConstantMap[C] = ConstantVector.size();
            ConstantVector.push_back(C);
            ++NumPromoted;
          }
        }
      }
    }
  }

  // Collect uses from PHI nodes after other uses, because when transforming
  // the function, we handle PHI uses afterwards.
  for (BasicBlock &BB : F) {
    for (PHINode &PN : BB.phis()) {
      for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
        if (auto *C = dyn_cast<ConstantFP>(PN.getIncomingValue(i))) {
          if (C->getType()->isDoubleTy()) {
            if (TLI->isFPImmLegal(C->getValueAPF(), MVT::f64, /*ForCodeSize*/ false))
              continue;
            UsesInFunc[C].push_back(&PN.getOperandUse(i));
            if (ConstantMap.find(C) == ConstantMap.end()) {
              ConstantMap[C] = ConstantVector.size();
              ConstantVector.push_back(C);
              ++NumPromoted;
            }
          }
        }
      }
    }
  }

  // Bail out if no promotable constants found.
  if (ConstantVector.empty())
    return false;

  // Create a global array containing the promoted constants.
  Module *M = F.getParent();
  Type *DoubleTy = Type::getDoubleTy(M->getContext());
  ArrayType *ArrayTy = ArrayType::get(DoubleTy, ConstantVector.size());
  Constant *GlobalArrayInitializer = ConstantArray::get(ArrayTy, ConstantVector);

  auto *GlobalArray = new GlobalVariable(
      *M, ArrayTy,
      /*isConstant=*/true, GlobalValue::InternalLinkage, GlobalArrayInitializer,
      ".promoted_doubles." + F.getName());

  // Create GEP for the base pointer in the function entry.
  IRBuilder<> EntryBuilder(&F.getEntryBlock().front());
  Value *BasePtr = EntryBuilder.CreateConstInBoundsGEP2_64(
      GlobalArray->getValueType(), GlobalArray, 0, 0, "doubles.base");

  // A cache to hold the loaded value for a given constant within a basic block.
  DenseMap<std::pair<ConstantFP *, BasicBlock *>, Value *> LocalLoads;

  // Replace all uses with the loaded value.
  for (Constant *ConstVal : ConstantVector) {
    auto *Const = cast<ConstantFP>(ConstVal);
    const auto &Uses = UsesInFunc.at(Const);
    unsigned Idx = ConstantMap.at(Const);

    for (Use *U : Uses) {
      Instruction *UserInst = cast<Instruction>(U->getUser());
      BasicBlock *InsertionBB;
      Instruction *InsertionPt;

      if (auto *PN = dyn_cast<PHINode>(UserInst)) {
        // If the user is a PHI node, we must insert the load in the
        // corresponding predecessor basic block, before its terminator.
        unsigned OperandIdx = U->getOperandNo();
        InsertionBB = PN->getIncomingBlock(OperandIdx);
        InsertionPt = InsertionBB->getTerminator();
      } else {
        // For any other instruction, we can insert the load right before it.
        InsertionBB = UserInst->getParent();
        InsertionPt = UserInst;
      }

      auto CacheKey = std::make_pair(Const, InsertionBB);
      Value *LoadedVal = nullptr;

      // Re-use a load if it exists in the insertion block.
      if (LocalLoads.count(CacheKey)) {
        LoadedVal = LocalLoads.at(CacheKey);
      } else {
        // Otherwise, create a new GEP and Load at the correct insertion point.
        IRBuilder<> Builder(InsertionPt);
        Value *ElementPtr = Builder.CreateConstInBoundsGEP1_64(
            DoubleTy, BasePtr, Idx, "double.addr");
        LoadedVal = Builder.CreateLoad(DoubleTy, ElementPtr, "double.val");

        // Cache the newly created load for this block.
        LocalLoads[CacheKey] = LoadedVal;
      }

      U->set(LoadedVal);
      ++NumPromotedUses;
    }
  }

  return true;
}
