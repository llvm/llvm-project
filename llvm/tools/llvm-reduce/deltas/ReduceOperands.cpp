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
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"

using namespace llvm;

static void
extractOperandsFromModule(Oracle &O, Module &Program,
                          function_ref<Value *(Use &)> ReduceValue) {
  for (auto &F : Program.functions()) {
    for (auto &I : instructions(&F)) {
      for (auto &Op : I.operands()) {
        Value *Reduced = ReduceValue(Op);
        if (Reduced && !O.shouldKeep())
          Op.set(Reduced);
      }
    }
  }
}

static int countOperands(Module &Program,
                         function_ref<Value *(Use &)> ReduceValue) {
  int Count = 0;
  for (auto &F : Program.functions()) {
    for (auto &I : instructions(&F)) {
      for (auto &Op : I.operands()) {
        if (ReduceValue(Op))
          Count++;
      }
    }
  }
  return Count;
}

static bool isOne(Use &Op) {
  auto *C = dyn_cast<Constant>(Op);
  return C && C->isOneValue();
}

static bool isZero(Use &Op) {
  auto *C = dyn_cast<Constant>(Op);
  return C && C->isNullValue();
}

void llvm::reduceOperandsUndefDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Operands to undef...\n";
  auto ReduceValue = [](Use &Op) -> Value * {
    if (isa<GEPOperator>(Op.getUser()))
      return nullptr;
    if (Op->getType()->isLabelTy())
      return nullptr;
    // Don't replace existing ConstantData Uses.
    return isa<ConstantData>(*Op) ? nullptr : UndefValue::get(Op->getType());
  };
  int Count = countOperands(Test.getProgram(), ReduceValue);
  runDeltaPass(Test, Count, [ReduceValue](Oracle &O, Module &Program) {
    extractOperandsFromModule(O, Program, ReduceValue);
  });
}

void llvm::reduceOperandsOneDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Operands to one...\n";
  auto ReduceValue = [](Use &Op) -> Value * {
    // TODO: support floats
    if (isa<GEPOperator>(Op.getUser()))
      return nullptr;
    auto *Ty = dyn_cast<IntegerType>(Op->getType());
    if (!Ty)
      return nullptr;
    // Don't replace existing ones and zeroes.
    return (isOne(Op) || isZero(Op)) ? nullptr : ConstantInt::get(Ty, 1);
  };
  int Count = countOperands(Test.getProgram(), ReduceValue);
  runDeltaPass(Test, Count, [ReduceValue](Oracle &O, Module &Program) {
    extractOperandsFromModule(O, Program, ReduceValue);
  });
}

void llvm::reduceOperandsZeroDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Operands to zero...\n";
  auto ReduceValue = [](Use &Op) -> Value * {
    // TODO: be more precise about which GEP operands we can reduce (e.g. array
    // indexes)
    if (isa<GEPOperator>(Op.getUser()))
      return nullptr;
    if (Op->getType()->isLabelTy())
      return nullptr;
    // Don't replace existing zeroes.
    return isZero(Op) ? nullptr : Constant::getNullValue(Op->getType());
  };
  int Count = countOperands(Test.getProgram(), ReduceValue);
  runDeltaPass(Test, Count, [ReduceValue](Oracle &O, Module &Program) {
    extractOperandsFromModule(O, Program, ReduceValue);
  });
}
