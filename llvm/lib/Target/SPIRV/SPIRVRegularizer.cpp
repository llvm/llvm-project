//===-- SPIRVRegularizer.cpp - regularize IR for SPIR-V ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements regularization of LLVM IR for SPIR-V. The prototype of
// the pass was taken from SPIRV-LLVM translator.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

#include <list>

#define DEBUG_TYPE "spirv-regularizer"

using namespace llvm;

namespace {
struct SPIRVRegularizer : public FunctionPass {
public:
  static char ID;
  SPIRVRegularizer() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return "SPIR-V Regularizer"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
  }

private:
  void runLowerConstExpr(Function &F);
  void runLowerI1Comparisons(Function &F);
};
} // namespace

char SPIRVRegularizer::ID = 0;

INITIALIZE_PASS(SPIRVRegularizer, DEBUG_TYPE, "SPIR-V Regularizer", false,
                false)

// Since SPIR-V cannot represent constant expression, constant expressions
// in LLVM IR need to be lowered to instructions. For each function,
// the constant expressions used by instructions of the function are replaced
// by instructions placed in the entry block since it dominates all other BBs.
// Each constant expression only needs to be lowered once in each function
// and all uses of it by instructions in that function are replaced by
// one instruction.
// TODO: remove redundant instructions for common subexpression.
void SPIRVRegularizer::runLowerConstExpr(Function &F) {
  LLVMContext &Ctx = F.getContext();
  std::list<Instruction *> WorkList;
  for (auto &II : instructions(F))
    WorkList.push_back(&II);

  auto FBegin = F.begin();
  while (!WorkList.empty()) {
    Instruction *II = WorkList.front();

    auto LowerOp = [&II, &FBegin, &F](Value *V) -> Value * {
      if (isa<Function>(V))
        return V;
      auto *CE = cast<ConstantExpr>(V);
      LLVM_DEBUG(dbgs() << "[lowerConstantExpressions] " << *CE);
      auto ReplInst = CE->getAsInstruction();
      auto InsPoint = II->getParent() == &*FBegin ? II : &FBegin->back();
      ReplInst->insertBefore(InsPoint->getIterator());
      LLVM_DEBUG(dbgs() << " -> " << *ReplInst << '\n');
      std::vector<Instruction *> Users;
      // Do not replace use during iteration of use. Do it in another loop.
      for (auto U : CE->users()) {
        LLVM_DEBUG(dbgs() << "[lowerConstantExpressions] Use: " << *U << '\n');
        auto InstUser = dyn_cast<Instruction>(U);
        // Only replace users in scope of current function.
        if (InstUser && InstUser->getParent()->getParent() == &F)
          Users.push_back(InstUser);
      }
      for (auto &User : Users) {
        if (ReplInst->getParent() == User->getParent() &&
            User->comesBefore(ReplInst))
          ReplInst->moveBefore(User->getIterator());
        User->replaceUsesOfWith(CE, ReplInst);
      }
      return ReplInst;
    };

    WorkList.pop_front();
    auto LowerConstantVec = [&II, &LowerOp, &WorkList,
                             &Ctx](ConstantVector *Vec,
                                   unsigned NumOfOp) -> Value * {
      if (std::all_of(Vec->op_begin(), Vec->op_end(), [](Value *V) {
            return isa<ConstantExpr>(V) || isa<Function>(V);
          })) {
        // Expand a vector of constexprs and construct it back with
        // series of insertelement instructions.
        std::list<Value *> OpList;
        std::transform(Vec->op_begin(), Vec->op_end(),
                       std::back_inserter(OpList),
                       [LowerOp](Value *V) { return LowerOp(V); });
        Value *Repl = nullptr;
        unsigned Idx = 0;
        auto *PhiII = dyn_cast<PHINode>(II);
        Instruction *InsPoint =
            PhiII ? &PhiII->getIncomingBlock(NumOfOp)->back() : II;
        std::list<Instruction *> ReplList;
        for (auto V : OpList) {
          if (auto *Inst = dyn_cast<Instruction>(V))
            ReplList.push_back(Inst);
          Repl = InsertElementInst::Create(
              (Repl ? Repl : PoisonValue::get(Vec->getType())), V,
              ConstantInt::get(Type::getInt32Ty(Ctx), Idx++), "",
              InsPoint->getIterator());
        }
        WorkList.splice(WorkList.begin(), ReplList);
        return Repl;
      }
      return nullptr;
    };
    for (unsigned OI = 0, OE = II->getNumOperands(); OI != OE; ++OI) {
      auto *Op = II->getOperand(OI);
      if (auto *Vec = dyn_cast<ConstantVector>(Op)) {
        Value *ReplInst = LowerConstantVec(Vec, OI);
        if (ReplInst)
          II->replaceUsesOfWith(Op, ReplInst);
      } else if (auto CE = dyn_cast<ConstantExpr>(Op)) {
        WorkList.push_front(cast<Instruction>(LowerOp(CE)));
      } else if (auto MDAsVal = dyn_cast<MetadataAsValue>(Op)) {
        auto ConstMD = dyn_cast<ConstantAsMetadata>(MDAsVal->getMetadata());
        if (!ConstMD)
          continue;
        Constant *C = ConstMD->getValue();
        Value *ReplInst = nullptr;
        if (auto *Vec = dyn_cast<ConstantVector>(C))
          ReplInst = LowerConstantVec(Vec, OI);
        if (auto *CE = dyn_cast<ConstantExpr>(C))
          ReplInst = LowerOp(CE);
        if (!ReplInst)
          continue;
        Metadata *RepMD = ValueAsMetadata::get(ReplInst);
        Value *RepMDVal = MetadataAsValue::get(Ctx, RepMD);
        II->setOperand(OI, RepMDVal);
        WorkList.push_front(cast<Instruction>(ReplInst));
      }
    }
  }
}

// Lower i1 comparisons with certain predicates to logical operations.
// The backend treats i1 as boolean values, and SPIR-V only allows logical
// operations for boolean values. This function lowers i1 comparisons with
// certain predicates to logical operations to generate valid SPIR-V.
void SPIRVRegularizer::runLowerI1Comparisons(Function &F) {
  for (auto &I : make_early_inc_range(instructions(F))) {
    auto *Cmp = dyn_cast<ICmpInst>(&I);
    if (!Cmp)
      continue;

    bool IsI1 = Cmp->getOperand(0)->getType()->isIntegerTy(1);
    if (!IsI1)
      continue;

    auto Pred = Cmp->getPredicate();
    bool IsTargetPred =
        Pred >= ICmpInst::ICMP_UGT && Pred <= ICmpInst::ICMP_SLE;
    if (!IsTargetPred)
      continue;

    Value *P = Cmp->getOperand(0);
    Value *Q = Cmp->getOperand(1);

    IRBuilder<> Builder(Cmp);
    Value *Result = nullptr;
    switch (Pred) {
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_SLT:
      // Result = p & !q
      Result = Builder.CreateAnd(P, Builder.CreateNot(Q));
      break;
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_SGT:
      // Result = q & !p
      Result = Builder.CreateAnd(Q, Builder.CreateNot(P));
      break;
    case ICmpInst::ICMP_ULE:
    case ICmpInst::ICMP_SGE:
      // Result = q | !p
      Result = Builder.CreateOr(Q, Builder.CreateNot(P));
      break;
    case ICmpInst::ICMP_UGE:
    case ICmpInst::ICMP_SLE:
      // Result = p | !q
      Result = Builder.CreateOr(P, Builder.CreateNot(Q));
      break;
    default:
      llvm_unreachable("Unexpected predicate");
    }

    Result->takeName(Cmp);
    Cmp->replaceAllUsesWith(Result);
    Cmp->eraseFromParent();
  }
}

bool SPIRVRegularizer::runOnFunction(Function &F) {
  runLowerI1Comparisons(F);
  runLowerConstExpr(F);
  return true;
}

FunctionPass *llvm::createSPIRVRegularizerPass() {
  return new SPIRVRegularizer();
}
