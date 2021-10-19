//===- llvm/IR/OptBisect/Bisect.cpp - LLVM Bisect support -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements support for a bisecting optimizations based on a
/// command line option.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/OptBisect.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <limits>

using namespace llvm;

static cl::opt<int> OptBisectLimit("opt-bisect-limit", cl::Hidden,
                                   cl::init(OptBisect::Disabled), cl::Optional,
                                   cl::cb<void, int>([](int Limit) {
                                     llvm::OptBisector->setLimit(Limit);
                                   }),
                                   cl::desc("Maximum optimization to perform"));

static void printPassMessage(const StringRef &Name, int PassNum,
                             StringRef TargetDesc, bool Running) {
  StringRef Status = Running ? "" : "NOT ";
  errs() << "BISECT: " << Status << "running pass "
         << "(" << PassNum << ") " << Name << " on " << TargetDesc << "\n";
}

bool OptBisect::shouldRunPass(const Pass *P, StringRef IRDescription) {
  assert(isEnabled());

  return checkPass(P->getPassName(), IRDescription);
}

bool OptBisect::checkPass(const StringRef PassName,
                          const StringRef TargetDesc) {
  assert(isEnabled());

  int CurBisectNum = ++LastBisectNum;
  bool ShouldRun = (BisectLimit == -1 || CurBisectNum <= BisectLimit);
  printPassMessage(PassName, CurBisectNum, TargetDesc, ShouldRun);
  return ShouldRun;
}

const int OptBisect::Disabled;

ManagedStatic<OptBisect> llvm::OptBisector;
