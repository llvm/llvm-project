//===------ FlattenSchedule.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Try to reduce the number of scatter dimension. Useful to make isl_union_map
// schedules more understandable. This is only intended for debugging and
// unittests, not for production use.
//
//===----------------------------------------------------------------------===//

#include "polly/FlattenSchedule.h"
#include "polly/FlattenAlgo.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ISLOStream.h"
#include "polly/Support/ISLTools.h"
#define DEBUG_TYPE "polly-flatten-schedule"

using namespace polly;
using namespace llvm;

namespace {

/// Print a schedule to @p OS.
///
/// Prints the schedule for each statements on a new line.
void printSchedule(raw_ostream &OS, const isl::union_map &Schedule,
                   int indent) {
  for (isl::map Map : Schedule.get_map_list())
    OS.indent(indent) << Map << "\n";
}

/// Flatten the schedule stored in an polly::Scop.
class FlattenSchedule final : public ScopPass {
private:
  FlattenSchedule(const FlattenSchedule &) = delete;
  const FlattenSchedule &operator=(const FlattenSchedule &) = delete;

  std::shared_ptr<isl_ctx> IslCtx;
  isl::union_map OldSchedule;

public:
  static char ID;
  explicit FlattenSchedule() : ScopPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitive<ScopInfoRegionPass>();
    AU.setPreservesAll();
  }

  bool runOnScop(Scop &S) override {
    // Keep a reference to isl_ctx to ensure that it is not freed before we free
    // OldSchedule.
    IslCtx = S.getSharedIslCtx();

    LLVM_DEBUG(dbgs() << "Going to flatten old schedule:\n");
    OldSchedule = S.getSchedule();
    LLVM_DEBUG(printSchedule(dbgs(), OldSchedule, 2));

    auto Domains = S.getDomains();
    auto RestrictedOldSchedule = OldSchedule.intersect_domain(Domains);
    LLVM_DEBUG(dbgs() << "Old schedule with domains:\n");
    LLVM_DEBUG(printSchedule(dbgs(), RestrictedOldSchedule, 2));

    auto NewSchedule = flattenSchedule(RestrictedOldSchedule);

    LLVM_DEBUG(dbgs() << "Flattened new schedule:\n");
    LLVM_DEBUG(printSchedule(dbgs(), NewSchedule, 2));

    NewSchedule = NewSchedule.gist_domain(Domains);
    LLVM_DEBUG(dbgs() << "Gisted, flattened new schedule:\n");
    LLVM_DEBUG(printSchedule(dbgs(), NewSchedule, 2));

    S.setSchedule(NewSchedule);
    return false;
  }

  void printScop(raw_ostream &OS, Scop &S) const override {
    OS << "Schedule before flattening {\n";
    printSchedule(OS, OldSchedule, 4);
    OS << "}\n\n";

    OS << "Schedule after flattening {\n";
    printSchedule(OS, S.getSchedule(), 4);
    OS << "}\n";
  }

  void releaseMemory() override {
    OldSchedule = {};
    IslCtx.reset();
  }
};

char FlattenSchedule::ID;

/// Print result from FlattenSchedule.
class FlattenSchedulePrinterLegacyPass final : public ScopPass {
public:
  static char ID;

  FlattenSchedulePrinterLegacyPass()
      : FlattenSchedulePrinterLegacyPass(outs()){};
  explicit FlattenSchedulePrinterLegacyPass(llvm::raw_ostream &OS)
      : ScopPass(ID), OS(OS) {}

  bool runOnScop(Scop &S) override {
    FlattenSchedule &P = getAnalysis<FlattenSchedule>();

    OS << "Printing analysis '" << P.getPassName() << "' for region: '"
       << S.getRegion().getNameStr() << "' in function '"
       << S.getFunction().getName() << "':\n";
    P.printScop(OS, S);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ScopPass::getAnalysisUsage(AU);
    AU.addRequired<FlattenSchedule>();
    AU.setPreservesAll();
  }

private:
  llvm::raw_ostream &OS;
};

char FlattenSchedulePrinterLegacyPass::ID = 0;
} // anonymous namespace

Pass *polly::createFlattenSchedulePass() { return new FlattenSchedule(); }

Pass *polly::createFlattenSchedulePrinterLegacyPass(llvm::raw_ostream &OS) {
  return new FlattenSchedulePrinterLegacyPass(OS);
}

INITIALIZE_PASS_BEGIN(FlattenSchedule, "polly-flatten-schedule",
                      "Polly - Flatten schedule", false, false)
INITIALIZE_PASS_END(FlattenSchedule, "polly-flatten-schedule",
                    "Polly - Flatten schedule", false, false)

INITIALIZE_PASS_BEGIN(FlattenSchedulePrinterLegacyPass,
                      "polly-print-flatten-schedule",
                      "Polly - Print flattened schedule", false, false)
INITIALIZE_PASS_DEPENDENCY(FlattenSchedule)
INITIALIZE_PASS_END(FlattenSchedulePrinterLegacyPass,
                    "polly-print-flatten-schedule",
                    "Polly - Print flattened schedule", false, false)
