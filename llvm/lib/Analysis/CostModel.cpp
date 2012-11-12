//===- CostModel.cpp ------ Cost Model Analysis ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the cost model analysis. It provides a very basic cost
// estimation for LLVM-IR. The cost result can be thought of as cycles, but it
// is really unit-less. The estimated cost is ment to be used for comparing
// alternatives.
//
//===----------------------------------------------------------------------===//

#define CM_NAME "cost-model"
#define DEBUG_TYPE CM_NAME
#include "llvm/Analysis/Passes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/TargetTransformInfo.h"
#include "llvm/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  class CostModelAnalysis : public FunctionPass {

  public:
    static char ID; // Class identification, replacement for typeinfo
    CostModelAnalysis() : FunctionPass(ID), F(0), VTTI(0) {
      initializeCostModelAnalysisPass(
        *PassRegistry::getPassRegistry());
    }

    /// Returns the expected cost of the instruction.
    /// Returns -1 if the cost is unknown.
    /// Note, this method does not cache the cost calculation and it
    /// can be expensive in some cases.
    unsigned getInstructionCost(Instruction *I) const;

  private:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool runOnFunction(Function &F);
    virtual void print(raw_ostream &OS, const Module*) const;

    /// The function that we analyze.
    Function *F;
    /// Vector target information.
    const VectorTargetTransformInfo *VTTI;
  };
}  // End of anonymous namespace

// Register this pass.
char CostModelAnalysis::ID = 0;
static const char cm_name[] = "Cost Model Analysis";
INITIALIZE_PASS_BEGIN(CostModelAnalysis, CM_NAME, cm_name, false, true)
INITIALIZE_PASS_END  (CostModelAnalysis, CM_NAME, cm_name, false, true)

FunctionPass *llvm::createCostModelAnalysisPass() {
  return new CostModelAnalysis();
}

void
CostModelAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool
CostModelAnalysis::runOnFunction(Function &F) {
 this->F = &F;

 // Target information.
 TargetTransformInfo *TTI;
 TTI = getAnalysisIfAvailable<TargetTransformInfo>();
 if (TTI)
   VTTI = TTI->getVectorTargetTransformInfo();

 return false;
}

unsigned CostModelAnalysis::getInstructionCost(Instruction *I) const {
  if (!VTTI)
    return -1;

  switch (I->getOpcode()) {
  case Instruction::Ret:
  case Instruction::PHI:
  case Instruction::Br: {
    return VTTI->getCFInstrCost(I->getOpcode());
  }
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    return VTTI->getArithmeticInstrCost(I->getOpcode(), I->getType());
  }
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    Type *CondTy = SI->getCondition()->getType();
    return VTTI->getCmpSelInstrCost(I->getOpcode(), I->getType(), CondTy);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();
    return VTTI->getCmpSelInstrCost(I->getOpcode(), ValTy);
  }
  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(I);
    Type *ValTy = SI->getValueOperand()->getType();
    return VTTI->getMemoryOpCost(I->getOpcode(), ValTy,
                                 SI->getAlignment(),
                                 SI->getPointerAddressSpace());
  }
  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(I);
    return VTTI->getMemoryOpCost(I->getOpcode(), I->getType(),
                                 LI->getAlignment(),
                                 LI->getPointerAddressSpace());
  }
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::BitCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    return VTTI->getCastInstrCost(I->getOpcode(), I->getType(), SrcTy);
  }
  case Instruction::ExtractElement: {
    ExtractElementInst * EEI = cast<ExtractElementInst>(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1));
    unsigned Idx = -1;
    if (CI)
      Idx = CI->getZExtValue();
    return VTTI->getVectorInstrCost(I->getOpcode(),
                                    EEI->getOperand(0)->getType(), Idx);
  }
  case Instruction::InsertElement: {
      InsertElementInst * IE = cast<InsertElementInst>(I);
      ConstantInt *CI = dyn_cast<ConstantInt>(IE->getOperand(2));
      unsigned Idx = -1;
      if (CI)
        Idx = CI->getZExtValue();
      return VTTI->getVectorInstrCost(I->getOpcode(),
                                      IE->getType(), Idx);
    }
  default:
    // We don't have any information on this instruction.
    return -1;
  }
}

void CostModelAnalysis::print(raw_ostream &OS, const Module*) const {
  if (!F)
    return;

  for (Function::iterator B = F->begin(), BE = F->end(); B != BE; ++B) {
    for (BasicBlock::iterator it = B->begin(), e = B->end(); it != e; ++it) {
      Instruction *Inst = it;
      unsigned Cost = getInstructionCost(Inst);
      if (Cost != (unsigned)-1)
        OS << "Cost Model: Found an estimated cost of " << Cost;
      else
        OS << "Cost Model: Unknown cost";

      OS << " for instruction: "<< *Inst << "\n";
    }
  }
}
