//===--- FunctionComparatorIgnoringConst.cpp - Function Comparator --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FunctionComparatorIgnoringConst.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/MergeFunctionsIgnoringConst.h"

using namespace llvm;

/// Returns true if the \OpIdx operand of \p CI is the callee operand.
static bool isCalleeOperand(const CallBase *CI, unsigned OpIdx) {
  return &CI->getCalledOperandUse() == &CI->getOperandUse(OpIdx);
}

bool isEligibleInstrunctionForConstantSharing(const Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Load:
  case Instruction::Store:
  case Instruction::Call:
    return true;
  default: {
    return false;
  }
  }
}

static bool canParameterizeCallOperand(const CallBase *CI, unsigned OpIdx) {
  if (CI->isInlineAsm())
    return false;
  Function *Callee = CI->getCalledOperand()
                         ? dyn_cast_or_null<Function>(
                               CI->getCalledOperand()->stripPointerCasts())
                         : nullptr;
  if (Callee) {
    if (Callee->isIntrinsic())
      return false;
    // objc_msgSend stubs must be called, and can't have their address taken.
    if (Callee->getName().startswith("objc_msgSend$"))
      return false;
  }
  if (isCalleeOperand(CI, OpIdx) &&
      CI->getOperandBundle(LLVMContext::OB_ptrauth).has_value()) {
    // The operand is the callee and it has already been signed. Ignore this
    // because we cannot add another ptrauth bundle to the call instruction.
    return false;
  }
  return true;
}

bool isEligibleOperandForConstantSharing(const Instruction *I, unsigned OpIdx) {
  assert(OpIdx < I->getNumOperands() && "Invalid operand index");

  if (!isEligibleInstrunctionForConstantSharing(I))
    return false;

  auto Opnd = I->getOperand(OpIdx);
  if (!isa<Constant>(Opnd))
    return false;

  if (const auto *CI = dyn_cast<CallBase>(I))
    return canParameterizeCallOperand(CI, OpIdx);

  return true;
}

int FunctionComparatorIgnoringConst::cmpOperandsIgnoringConsts(
    const Instruction *L, const Instruction *R, unsigned opIdx) {
  Value *OpL = L->getOperand(opIdx);
  Value *OpR = R->getOperand(opIdx);

  int Res = cmpValues(OpL, OpR);
  if (Res == 0)
    return Res;

  if (!isa<Constant>(OpL) || !isa<Constant>(OpR))
    return Res;

  if (!isEligibleOperandForConstantSharing(L, opIdx) ||
      !isEligibleOperandForConstantSharing(R, opIdx))
    return Res;

  if (cmpTypes(OpL->getType(), OpR->getType()))
    return Res;

  return 0;
}

// Test whether two basic blocks have equivalent behavior.
int FunctionComparatorIgnoringConst::cmpBasicBlocksIgnoringConsts(
    const BasicBlock *BBL, const BasicBlock *BBR,
    const std::set<std::pair<int, int>> *InstOpndIndex) {
  BasicBlock::const_iterator InstL = BBL->begin(), InstLE = BBL->end();
  BasicBlock::const_iterator InstR = BBR->begin(), InstRE = BBR->end();

  do {
    bool needToCmpOperands = true;
    if (int Res = cmpOperations(&*InstL, &*InstR, needToCmpOperands))
      return Res;
    if (needToCmpOperands) {
      assert(InstL->getNumOperands() == InstR->getNumOperands());

      for (unsigned i = 0, e = InstL->getNumOperands(); i != e; ++i) {
        // When a set for (instruction, operand) index pairs is given, we only
        // ignore constants located at such indices. Otherwise, we precisely
        // compare the operands.
        if (InstOpndIndex && !InstOpndIndex->count(std::make_pair(Index, i))) {
          Value *OpL = InstL->getOperand(i);
          Value *OpR = InstR->getOperand(i);
          if (int Res = cmpValues(OpL, OpR))
            return Res;
        }
        if (int Res = cmpOperandsIgnoringConsts(&*InstL, &*InstR, i))
          return Res;
        // cmpValues should ensure this is true.
        assert(cmpTypes(InstL->getOperand(i)->getType(),
                        InstR->getOperand(i)->getType()) == 0);
      }
    }
    ++Index;
    ++InstL, ++InstR;
  } while (InstL != InstLE && InstR != InstRE);

  if (InstL != InstLE && InstR == InstRE)
    return 1;
  if (InstL == InstLE && InstR != InstRE)
    return -1;
  return 0;
}

// Test whether the two functions have equivalent behavior.
int FunctionComparatorIgnoringConst::compareIgnoringConsts(
    const std::set<std::pair<int, int>> *InstOpndIndex) {
  beginCompare();
  Index = 0;

  if (int Res = compareSignature())
    return Res;

  Function::const_iterator LIter = FnL->begin(), LEnd = FnL->end();
  Function::const_iterator RIter = FnR->begin(), REnd = FnR->end();

  do {
    const BasicBlock *BBL = &*LIter;
    const BasicBlock *BBR = &*RIter;

    if (int Res = cmpValues(BBL, BBR))
      return Res;

    if (int Res = cmpBasicBlocksIgnoringConsts(BBL, BBR, InstOpndIndex))
      return Res;

    ++LIter, ++RIter;
  } while (LIter != LEnd && RIter != REnd);

  return 0;
}
