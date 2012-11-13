//===- Dependency.cpp - Calculate dependency information for a Scop.  -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Calculate the data dependency relations for a Scop using ISL.
//
// The integer set library (ISL) from Sven, has a integrated dependency analysis
// to calculate data dependences. This pass takes advantage of this and
// calculate those dependences a Scop.
//
// The dependences in this pass are exact in terms that for a specific read
// statement instance only the last write statement instance is returned. In
// case of may writes a set of possible write instances is returned. This
// analysis will never produce redundant dependences.
//
//===----------------------------------------------------------------------===//
//
#include "polly/Dependences.h"

#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"

#define DEBUG_TYPE "polly-dependences"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"

#include <isl/aff.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/set.h>

using namespace polly;
using namespace llvm;

static cl::opt<bool>
  LegalityCheckDisabled("disable-polly-legality",
       cl::desc("Disable polly legality check"), cl::Hidden,
       cl::init(false));

static cl::opt<bool>
  ValueDependences("polly-value-dependences",
       cl::desc("Use value instead of memory based dependences"), cl::Hidden,
       cl::init(true));

//===----------------------------------------------------------------------===//
Dependences::Dependences() : ScopPass(ID) {
  RAW = WAR = WAW = NULL;
}

void Dependences::collectInfo(Scop &S,
                              isl_union_map **Read, isl_union_map **Write,
                              isl_union_map **MayWrite,
                              isl_union_map **Schedule) {
  isl_space *Space = S.getParamSpace();
  *Read = isl_union_map_empty(isl_space_copy(Space));
  *Write = isl_union_map_empty(isl_space_copy(Space));
  *MayWrite = isl_union_map_empty(isl_space_copy(Space));
  *Schedule = isl_union_map_empty(Space);

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    for (ScopStmt::memacc_iterator MI = Stmt->memacc_begin(),
          ME = Stmt->memacc_end(); MI != ME; ++MI) {
      isl_set *domcp = Stmt->getDomain();
      isl_map *accdom = (*MI)->getAccessRelation();

      accdom = isl_map_intersect_domain(accdom, domcp);

      if ((*MI)->isRead())
        *Read = isl_union_map_add_map(*Read, accdom);
      else
        *Write = isl_union_map_add_map(*Write, accdom);
    }
    *Schedule = isl_union_map_add_map(*Schedule, Stmt->getScattering());
  }
}

void Dependences::calculateDependences(Scop &S) {
  isl_union_map *Read, *Write, *MayWrite, *Schedule;

  collectInfo(S, &Read, &Write, &MayWrite, &Schedule);

  if (ValueDependences) {
    isl_union_map_compute_flow(isl_union_map_copy(Read),
                               isl_union_map_copy(Write),
                               isl_union_map_copy(MayWrite),
                               isl_union_map_copy(Schedule),
                               &RAW, NULL, NULL, NULL);

    isl_union_map_compute_flow(isl_union_map_copy(Write),
                               isl_union_map_copy(Write),
                               isl_union_map_copy(Read),
                               isl_union_map_copy(Schedule),
                               &WAW, &WAR, NULL, NULL);
  } else {
    isl_union_map *Empty;

    Empty = isl_union_map_empty(isl_union_map_get_space(Write));
    Write = isl_union_map_union(Write, isl_union_map_copy(MayWrite));

    isl_union_map_compute_flow(isl_union_map_copy(Read),
                               isl_union_map_copy(Empty),
                               isl_union_map_copy(Write),
                               isl_union_map_copy(Schedule),
                               NULL, &RAW, NULL, NULL);

    isl_union_map_compute_flow(isl_union_map_copy(Write),
                               isl_union_map_copy(Empty),
                               isl_union_map_copy(Read),
                               isl_union_map_copy(Schedule),
                               NULL, &WAR, NULL, NULL);

    isl_union_map_compute_flow(isl_union_map_copy(Write),
                               isl_union_map_copy(Empty),
                               isl_union_map_copy(Write),
                               isl_union_map_copy(Schedule),
                               NULL, &WAW, NULL, NULL);
    isl_union_map_free(Empty);
  }

  isl_union_map_free(MayWrite);
  isl_union_map_free(Write);
  isl_union_map_free(Read);
  isl_union_map_free(Schedule);

  RAW = isl_union_map_coalesce(RAW);
  WAW = isl_union_map_coalesce(WAW);
  WAR = isl_union_map_coalesce(WAR);
}

bool Dependences::runOnScop(Scop &S) {
  releaseMemory();
  calculateDependences(S);

  return false;
}

bool Dependences::isValidScattering(StatementToIslMapTy *NewScattering) {
  Scop &S = getCurScop();

  if (LegalityCheckDisabled)
    return true;

  isl_union_map *Dependences = getDependences(TYPE_ALL);
  isl_space *Space = S.getParamSpace();
  isl_union_map *Scattering = isl_union_map_empty(Space);

  isl_space *ScatteringSpace = 0;

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    isl_map *StmtScat;

    if (NewScattering->find(*SI) == NewScattering->end())
      StmtScat = Stmt->getScattering();
    else
      StmtScat = isl_map_copy((*NewScattering)[Stmt]);

    if (!ScatteringSpace)
      ScatteringSpace = isl_space_range(isl_map_get_space(StmtScat));

    Scattering = isl_union_map_add_map(Scattering, StmtScat);
  }

  Dependences = isl_union_map_apply_domain(Dependences,
                                           isl_union_map_copy(Scattering));
  Dependences = isl_union_map_apply_range(Dependences, Scattering);

  isl_set *Zero = isl_set_universe(isl_space_copy(ScatteringSpace));
  for (unsigned i = 0; i < isl_set_dim(Zero, isl_dim_set); i++)
    Zero = isl_set_fix_si(Zero, isl_dim_set, i, 0);

  isl_union_set *UDeltas = isl_union_map_deltas(Dependences);
  isl_set *Deltas = isl_union_set_extract_set(UDeltas, ScatteringSpace);
  isl_union_set_free(UDeltas);

  isl_map *NonPositive = isl_set_lex_le_set(Deltas, Zero);
  bool IsValid = isl_map_is_empty(NonPositive);
  isl_map_free(NonPositive);

  return IsValid;
}

isl_union_map *getCombinedScheduleForSpace(Scop *scop, unsigned dimLevel) {
  isl_space *Space = scop->getParamSpace();
  isl_union_map *schedule = isl_union_map_empty(Space);

  for (Scop::iterator SI = scop->begin(), SE = scop->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    unsigned remainingDimensions = Stmt->getNumScattering() - dimLevel;
    isl_map *Scattering = isl_map_project_out(Stmt->getScattering(),
                                              isl_dim_out, dimLevel,
                                              remainingDimensions);
    schedule = isl_union_map_add_map(schedule, Scattering);
  }

  return schedule;
}

bool Dependences::isParallelDimension(__isl_take isl_set *ScheduleSubset,
                                      unsigned ParallelDim) {
  // To check if a loop is parallel, we perform the following steps:
  //
  // o Move dependences from 'Domain -> Domain' to 'Schedule -> Schedule' space.
  // o Limit dependences to the schedule space enumerated by the loop.
  // o Calculate distances of the dependences.
  // o Check if one of the distances is invalid in presence of parallelism.

  isl_union_map *Schedule, *Deps;
  isl_map *ScheduleDeps;
  Scop *S = &getCurScop();

  Deps = getDependences(TYPE_ALL);

  if (isl_union_map_is_empty(Deps)) {
    isl_union_map_free(Deps);
    isl_set_free(ScheduleSubset);
    return true;
  }

  Schedule = getCombinedScheduleForSpace(S, ParallelDim);
  Deps = isl_union_map_apply_range(Deps, isl_union_map_copy(Schedule));
  Deps = isl_union_map_apply_domain(Deps, Schedule);
  ScheduleDeps = isl_map_from_union_map(Deps);
  ScheduleDeps = isl_map_intersect_domain(ScheduleDeps,
                                          isl_set_copy(ScheduleSubset));
  ScheduleDeps = isl_map_intersect_range(ScheduleDeps, ScheduleSubset);

  isl_set *Distances = isl_map_deltas(ScheduleDeps);
  isl_space *Space = isl_set_get_space(Distances);
  isl_set *Invalid = isl_set_universe(Space);

  // [0, ..., 0, +] - All zeros and last dimension larger than zero
  for (unsigned i = 0; i < ParallelDim - 1; i++)
    Invalid = isl_set_fix_si(Invalid, isl_dim_set, i, 0);

  Invalid = isl_set_lower_bound_si(Invalid, isl_dim_set, ParallelDim - 1, 1);
  Invalid = isl_set_intersect(Invalid, Distances);

  bool IsParallel = isl_set_is_empty(Invalid);
  isl_set_free(Invalid);

  return IsParallel;
}

void Dependences::printScop(raw_ostream &OS) const {
  std::string RAWString, WARString, WAWString;

  RAWString = polly::stringFromIslObj(RAW);
  WARString = polly::stringFromIslObj(WAR);
  WAWString = polly::stringFromIslObj(WAW);

  OS << "\tRAW dependences:\n\t\t" << RAWString << "\n";
  OS << "\tWAR dependences:\n\t\t" << WARString << "\n";
  OS << "\tWAW dependences:\n\t\t" << WAWString << "\n";
}

void Dependences::releaseMemory() {
  isl_union_map_free(RAW);
  isl_union_map_free(WAR);
  isl_union_map_free(WAW);

  RAW = WAR = WAW  = NULL;
}

isl_union_map *Dependences::getDependences(int Kinds) {
  isl_space *Space = isl_union_map_get_space(RAW);
  isl_union_map *Deps = isl_union_map_empty(Space);

  if (Kinds & TYPE_RAW)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(RAW));

  if (Kinds & TYPE_WAR)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(WAR));

  if (Kinds & TYPE_WAW)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(WAW));

  Deps = isl_union_map_coalesce(Deps);
  Deps = isl_union_map_detect_equalities(Deps);
  return Deps;
}

void Dependences::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
}

char Dependences::ID = 0;

INITIALIZE_PASS_BEGIN(Dependences, "polly-dependences",
                      "Polly - Calculate dependences", false, false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_END(Dependences, "polly-dependences",
                    "Polly - Calculate dependences", false, false)

Pass *polly::createDependencesPass() {
  return new Dependences();
}
