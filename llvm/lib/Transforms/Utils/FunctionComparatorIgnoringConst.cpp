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
#include "llvm/Transforms/Utils/MergeFunctionsIgnoringConst.h"

using namespace llvm;

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
