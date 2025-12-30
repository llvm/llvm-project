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

using namespace llvm;

static OptBisect &getOptBisector() {
  static OptBisect OptBisector;
  return OptBisector;
}

static OptDisable &getOptDisabler() {
  static OptDisable OptDisabler;
  return OptDisabler;
}

static cl::opt<int> OptBisectLimit("opt-bisect-limit", cl::Hidden,
                                   cl::init(OptBisect::Disabled), cl::Optional,
                                   cl::cb<void, int>([](int Limit) {
                                     getOptBisector().setLimit(Limit);
                                   }),
                                   cl::desc("Maximum optimization to perform"));

static cl::opt<bool> OptBisectVerbose(
    "opt-bisect-verbose",
    cl::desc("Show verbose output when opt-bisect-limit is set"), cl::Hidden,
    cl::init(true), cl::Optional);

static cl::list<std::string> OptDisablePasses(
    "opt-disable", cl::Hidden, cl::CommaSeparated, cl::Optional,
    cl::cb<void, std::string>([](const std::string &Pass) {
      getOptDisabler().setDisabled(Pass);
    }),
    cl::desc("Optimization pass(es) to disable (comma-separated list)"));

static cl::opt<bool>
    OptDisableVerbose("opt-disable-enable-verbosity",
                      cl::desc("Show verbose output when opt-disable is set"),
                      cl::Hidden, cl::init(false), cl::Optional);

static void printPassMessage(StringRef Name, int PassNum, StringRef TargetDesc,
                             bool Running) {
  StringRef Status = Running ? "" : "NOT ";
  errs() << "BISECT: " << Status << "running pass (" << PassNum << ") " << Name
         << " on " << TargetDesc << '\n';
}

bool OptBisect::shouldRunPass(StringRef PassName,
                              StringRef IRDescription) const {
  assert(isEnabled());

  int CurBisectNum = ++LastBisectNum;
  bool ShouldRun = (BisectLimit == -1 || CurBisectNum <= BisectLimit);
  if (OptBisectVerbose)
    printPassMessage(PassName, CurBisectNum, IRDescription, ShouldRun);
  return ShouldRun;
}

static void printDisablePassMessage(const StringRef &Name, StringRef TargetDesc,
                                    bool Running) {
  StringRef Status = Running ? "" : "NOT ";
  dbgs() << "OptDisable: " << Status << "running pass " << Name << " on "
         << TargetDesc << "\n";
}

void OptDisable::setDisabled(StringRef Pass) { DisabledPasses.insert(Pass); }

bool OptDisable::shouldRunPass(StringRef PassName,
                               StringRef IRDescription) const {
  assert(isEnabled());

  const bool ShouldRun = !DisabledPasses.contains(PassName);
  if (OptDisableVerbose)
    printDisablePassMessage(PassName, IRDescription, ShouldRun);
  return ShouldRun;
}

OptPassGate &llvm::getGlobalPassGate() {
  if (getOptDisabler().isEnabled())
    return getOptDisabler();
  return getOptBisector();
}
