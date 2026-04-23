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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IntegerInclusiveInterval.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>

using namespace llvm;

static OptBisect &getOptBisector() {
  static OptBisect OptBisector;
  return OptBisector;
}

static cl::opt<int> OptBisectLimit(
    "opt-bisect-limit", cl::Hidden, cl::init(-1), cl::Optional,
    cl::cb<void, int>([](int Limit) {
      if (Limit == -1)
        // -1 means run all passes.
        getOptBisector().setIntervals({{1, std::numeric_limits<int>::max()}});
      else if (Limit == 0)
        // 0 means run no passes.
        getOptBisector().setIntervals({{0, 0}});
      else if (Limit > 0)
        // Convert limit to interval 1-Limit.
        getOptBisector().setIntervals({{1, Limit}});
      else
        llvm_unreachable(
            ("Invalid limit for -opt-bisect-limit: " + llvm::utostr(Limit))
                .c_str());
    }),
    cl::desc(
        "Maximum optimization to perform (equivalent to -opt-bisect=1-N)"));

static cl::opt<std::string> OptBisectIntervals(
    "opt-bisect", cl::Hidden, cl::Optional,
    cl::cb<void, const std::string &>([](const std::string &IntervalStr) {
      if (IntervalStr == "-1") {
        // -1 means run all passes.
        getOptBisector().setIntervals({{1, std::numeric_limits<int>::max()}});
        return;
      }

      auto Intervals =
          IntegerInclusiveIntervalUtils::parseIntervals(IntervalStr);
      if (!Intervals) {
        handleAllErrors(Intervals.takeError(), [&](const StringError &E) {
          errs() << "Error: Invalid interval specification for -opt-bisect: "
                 << IntervalStr << " (" << E.getMessage() << ")\n";
        });
        exit(1);
      }
      getOptBisector().setIntervals(std::move(*Intervals));
    }),
    cl::desc("Run optimization passes only for the specified intervals. "
             "Format: '1-10,20-30,45' runs passes 1-10, 20-30, and 45, where "
             "index 1 is the first pass. Supply '0' to run no passes and -1 to "
             "run all passes."));

static cl::opt<bool> OptBisectVerbose(
    "opt-bisect-verbose",
    cl::desc(
        "Show verbose output when opt-bisect-limit and/or opt-disable are set"),
    cl::Hidden, cl::init(true), cl::Optional);

static cl::list<std::string> OptDisablePasses(
    "opt-disable", cl::Hidden, cl::CommaSeparated, cl::Optional,
    cl::cb<void, std::string>([](const std::string &Pass) {
      getOptBisector().setDisabled(Pass);
    }),
    cl::desc("Optimization pass(es) to disable (comma-separated list)"));

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

  // Check if current pass number falls within any of the specified intervals.
  // Since the bisector may be enabled by opt-disable, we also need to check if
  // the BisectIntervals are empty.
  bool ShouldRun =
      BisectIntervals.empty() ||
      IntegerInclusiveIntervalUtils::contains(BisectIntervals, CurBisectNum);

  // Also check if the pass is disabled via -opt-disable.
  ShouldRun = ShouldRun && !DisabledPasses.contains(PassName);

  if (OptBisectVerbose)
    printPassMessage(PassName, CurBisectNum, IRDescription, ShouldRun);
  return ShouldRun;
}

OptPassGate &llvm::getGlobalPassGate() { return getOptBisector(); }
