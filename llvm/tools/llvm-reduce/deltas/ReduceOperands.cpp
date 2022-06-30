//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReduceOperands.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"

using namespace llvm;
using namespace PatternMatch;

static void
extractOperandsFromModule(Oracle &O, Module &Program,
                          function_ref<Value *(Use &)> ReduceValue) {
  for (auto &F : Program.functions()) {
    for (auto &I : instructions(&F)) {
      for (auto &Op : I.operands()) {
        if (!O.shouldKeep()) {
          if (Value *Reduced = ReduceValue(Op))
            Op.set(Reduced);
        }
      }
    }
  }
}

static bool isOne(Use &Op) {
  auto *C = dyn_cast<Constant>(Op);
  return C && C->isOneValue();
}

static bool isZero(Use &Op) {
  auto *C = dyn_cast<Constant>(Op);
  return C && C->isNullValue();
}

static bool isZeroOrOneFP(Value *Op) {
  const APFloat *C;
  return match(Op, m_APFloat(C)) &&
         ((C->isZero() && !C->isNegative()) || C->isExactlyValue(1.0));
}

static bool shouldReduceOperand(Use &Op) {
  Type *Ty = Op->getType();
  if (Ty->isLabelTy() || Ty->isMetadataTy())
    return false;
  // TODO: be more precise about which GEP operands we can reduce (e.g. array
  // indexes)
  if (isa<GEPOperator>(Op.getUser()))
    return false;
  if (auto *CB = dyn_cast<CallBase>(Op.getUser())) {
    if (&CB->getCalledOperandUse() == &Op)
      return false;
  }
  return true;
}

void llvm::reduceOperandsOneDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Operands to one...\n";
  auto ReduceValue = [](Use &Op) -> Value * {
    if (!shouldReduceOperand(Op))
      return nullptr;

    Type *Ty = Op->getType();
    if (auto *IntTy = dyn_cast<IntegerType>(Ty)) {
      // Don't replace existing ones and zeroes.
      return (isOne(Op) || isZero(Op)) ? nullptr : ConstantInt::get(IntTy, 1);
    }

    if (Ty->isFloatingPointTy())
      return isZeroOrOneFP(Op) ? nullptr : ConstantFP::get(Ty, 1.0);

    if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
      if (isZeroOrOneFP(Op))
        return nullptr;

      return ConstantVector::getSplat(
          VT->getElementCount(), ConstantFP::get(VT->getElementType(), 1.0));
    }

    return nullptr;
  };
  runDeltaPass(Test, [ReduceValue](Oracle &O, Module &Program) {
    extractOperandsFromModule(O, Program, ReduceValue);
  });
}

void llvm::reduceOperandsZeroDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Operands to zero...\n";
  auto ReduceValue = [](Use &Op) -> Value * {
    if (!shouldReduceOperand(Op))
      return nullptr;
    // Don't replace existing zeroes.
    return isZero(Op) ? nullptr : Constant::getNullValue(Op->getType());
  };
  runDeltaPass(Test, [ReduceValue](Oracle &O, Module &Program) {
    extractOperandsFromModule(O, Program, ReduceValue);
  });
}

void llvm::reduceOperandsNaNDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Operands to NaN...\n";
  auto ReduceValue = [](Use &Op) -> Value * {
    Type *Ty = Op->getType();
    if (!Ty->isFPOrFPVectorTy())
      return nullptr;

    // Prefer 0.0 or 1.0 over NaN.
    //
    // TODO: Preferring NaN may make more sense because FP operations are more
    // universally foldable.
    if (match(Op.get(), m_NaN()) || isZeroOrOneFP(Op.get()))
      return nullptr;

    if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
      return ConstantVector::getSplat(VT->getElementCount(),
                                      ConstantFP::getQNaN(VT->getElementType()));
    }

    return ConstantFP::getQNaN(Ty);
  };
  runDeltaPass(Test, [ReduceValue](Oracle &O, Module &Program) {
    extractOperandsFromModule(O, Program, ReduceValue);
  });
}
