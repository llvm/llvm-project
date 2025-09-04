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
#include "llvm/Support/Range.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>

using namespace llvm;

static OptBisect &getOptBisector() {
  static OptBisect OptBisector;
  return OptBisector;
}

static OptDisable &getOptDisabler() {
  static OptDisable OptDisabler;
  return OptDisabler;
}

static cl::opt<int> OptBisectLimit(
    "opt-bisect-limit", cl::Hidden, cl::init(-1), cl::Optional,
    cl::cb<void, int>([](int Limit) {
      if (Limit == -1) {
        // -1 means run all passes
        getOptBisector().setRanges({{1, std::numeric_limits<int>::max()}});
      } else if (Limit == 0) {
        // 0 means run no passes
        getOptBisector().setRanges({{0, 0}});
      } else if (Limit > 0) {
        // Convert limit to range 1-Limit
        std::string RangeStr = Limit == 1 ? "1" : "1-" + llvm::utostr(Limit);
        auto Ranges = RangeUtils::parseRanges(RangeStr);
        if (!Ranges) {
          handleAllErrors(Ranges.takeError(), [&](const StringError &E) {
            errs() << "Error: Invalid limit for -opt-bisect-limit: " << Limit
                   << " (" << E.getMessage() << ")\n";
          });
          exit(1);
        }
        getOptBisector().setRanges(std::move(*Ranges));
      }
    }),
    cl::desc(
        "Maximum optimization to perform (equivalent to -opt-bisect=1-N)"));

static cl::opt<std::string> OptBisectRanges(
    "opt-bisect", cl::Hidden, cl::Optional,
    cl::cb<void, const std::string &>([](const std::string &RangeStr) {
      if (RangeStr == "-1") {
        // -1 means run all passes
        getOptBisector().setRanges({{1, std::numeric_limits<int>::max()}});
        return;
      }

      auto Ranges = RangeUtils::parseRanges(RangeStr);
      if (!Ranges) {
        handleAllErrors(Ranges.takeError(), [&](const StringError &E) {
          errs() << "Error: Invalid range specification for -opt-bisect: "
                 << RangeStr << " (" << E.getMessage() << ")\n";
        });
        exit(1);
      }
      getOptBisector().setRanges(std::move(*Ranges));
    }),
    cl::desc("Run optimization passes only for the specified ranges. "
             "Format: '1-10,20-30,45' (runs passes 1-10, 20-30, and 45). Pass '0' to run no passes and -1 to run all passes."));

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

  // Check if current pass number falls within any of the specified ranges
  bool ShouldRun = RangeUtils::contains(BisectRanges, CurBisectNum);

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
