//===------ FlattenSchedule.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#define DEBUG_TYPE "polly-flatten-schedule"

using namespace polly;
using namespace llvm;

namespace {

/// Print a schedule to @p OS.
///
/// Prints the schedule for each statements on a new line.
void printSchedule(raw_ostream &OS, const isl::union_map &Schedule,
                   int indent) {
  foreachElt(Schedule,
             [&OS, indent](isl::map Map) { OS.indent(indent) << Map << "\n"; });
}

/// Flatten the schedule stored in an polly::Scop.
class FlattenSchedule : public ScopPass {
private:
  FlattenSchedule(const FlattenSchedule &) = delete;
  const FlattenSchedule &operator=(const FlattenSchedule &) = delete;

  std::shared_ptr<isl_ctx> IslCtx;
  isl::union_map OldSchedule;

public:
  static char ID;
  explicit FlattenSchedule() : ScopPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitive<ScopInfoRegionPass>();
    AU.setPreservesAll();
  }

  virtual bool runOnScop(Scop &S) override {
    // Keep a reference to isl_ctx to ensure that it is not freed before we free
    // OldSchedule.
    IslCtx = S.getSharedIslCtx();

    DEBUG(dbgs() << "Going to flatten old schedule:\n");
    OldSchedule = give(S.getSchedule());
    DEBUG(printSchedule(dbgs(), OldSchedule, 2));

    auto Domains = give(S.getDomains());
    auto RestrictedOldSchedule = OldSchedule.intersect_domain(Domains);
    DEBUG(dbgs() << "Old schedule with domains:\n");
    DEBUG(printSchedule(dbgs(), RestrictedOldSchedule, 2));

    auto NewSchedule = flattenSchedule(RestrictedOldSchedule);

    DEBUG(dbgs() << "Flattened new schedule:\n");
    DEBUG(printSchedule(dbgs(), NewSchedule, 2));

    NewSchedule = NewSchedule.gist_domain(Domains);
    DEBUG(dbgs() << "Gisted, flattened new schedule:\n");
    DEBUG(printSchedule(dbgs(), NewSchedule, 2));

    S.setSchedule(NewSchedule.take());
    return false;
  }

  virtual void printScop(raw_ostream &OS, Scop &S) const override {
    OS << "Schedule before flattening {\n";
    printSchedule(OS, OldSchedule, 4);
    OS << "}\n\n";

    OS << "Schedule after flattening {\n";
    printSchedule(OS, give(S.getSchedule()), 4);
    OS << "}\n";
  }

  virtual void releaseMemory() override {
    OldSchedule = nullptr;
    IslCtx.reset();
  }
};

char FlattenSchedule::ID;
} // anonymous namespace

Pass *polly::createFlattenSchedulePass() { return new FlattenSchedule(); }

INITIALIZE_PASS_BEGIN(FlattenSchedule, "polly-flatten-schedule",
                      "Polly - Flatten schedule", false, false)
INITIALIZE_PASS_END(FlattenSchedule, "polly-flatten-schedule",
                    "Polly - Flatten schedule", false, false)
