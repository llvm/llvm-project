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
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/ISLOStream.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/PollyDebug.h"
#define DEBUG_TYPE "polly-flatten-schedule"

using namespace polly;
using namespace llvm;

namespace {

static cl::opt<bool> PollyPrintFlattenSchedule("polly-print-flatten-schedule",
                                               cl::desc("A polly pass"),
                                               cl::cat(PollyCategory));

/// Print a schedule to @p OS.
///
/// Prints the schedule for each statements on a new line.
void printSchedule(raw_ostream &OS, const isl::union_map &Schedule,
                   int indent) {
  for (isl::map Map : Schedule.get_map_list())
    OS.indent(indent) << Map << "\n";
}
} // namespace

void polly::runFlattenSchedulePass(Scop &S) {
  // Keep a reference to isl_ctx to ensure that it is not freed before we free
  // OldSchedule.
  auto IslCtx = S.getSharedIslCtx();

  POLLY_DEBUG(dbgs() << "Going to flatten old schedule:\n");
  auto OldSchedule = S.getSchedule();
  POLLY_DEBUG(printSchedule(dbgs(), OldSchedule, 2));

  auto Domains = S.getDomains();
  auto RestrictedOldSchedule = OldSchedule.intersect_domain(Domains);
  POLLY_DEBUG(dbgs() << "Old schedule with domains:\n");
  POLLY_DEBUG(printSchedule(dbgs(), RestrictedOldSchedule, 2));

  auto NewSchedule = flattenSchedule(RestrictedOldSchedule);

  POLLY_DEBUG(dbgs() << "Flattened new schedule:\n");
  POLLY_DEBUG(printSchedule(dbgs(), NewSchedule, 2));

  NewSchedule = NewSchedule.gist_domain(Domains);
  POLLY_DEBUG(dbgs() << "Gisted, flattened new schedule:\n");
  POLLY_DEBUG(printSchedule(dbgs(), NewSchedule, 2));

  S.setSchedule(NewSchedule);

  if (PollyPrintFlattenSchedule) {
    outs()
        << "Printing analysis 'Polly - Print flattened schedule' for region: '"
        << S.getRegion().getNameStr() << "' in function '"
        << S.getFunction().getName() << "':\n";

    outs() << "Schedule before flattening {\n";
    printSchedule(outs(), OldSchedule, 4);
    outs() << "}\n\n";

    outs() << "Schedule after flattening {\n";
    printSchedule(outs(), S.getSchedule(), 4);
    outs() << "}\n";
  }
}
