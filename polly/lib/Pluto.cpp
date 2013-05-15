//===- Pluto.cpp - Calculate an optimized schedule ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Use libpluto to optimize the schedule.
//
//===----------------------------------------------------------------------===//

#include "polly/Config/config.h"

#ifdef PLUTO_FOUND
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/Dependences.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"

#define DEBUG_TYPE "polly-opt-pluto"
#include "llvm/Support/Debug.h"

#include "pluto/libpluto.h"
#include "isl/map.h"

using namespace llvm;
using namespace polly;

static cl::opt<bool> EnableTiling("polly-pluto-tile", cl::desc("Enable tiling"),
                                  cl::Hidden, cl::init(false),
                                  cl::cat(PollyCategory));

namespace {
/// Convert an int into a string.
static std::string convertInt(int number) {
  if (number == 0)
    return "0";
  std::string temp = "";
  std::string returnvalue = "";
  while (number > 0) {
    temp += number % 10 + 48;
    number /= 10;
  }
  for (unsigned i = 0; i < temp.length(); i++)
    returnvalue += temp[temp.length() - i - 1];
  return returnvalue;
}

class PlutoOptimizer : public ScopPass {

public:
  static char ID;
  explicit PlutoOptimizer() : ScopPass(ID) {}

  virtual bool runOnScop(Scop &S);
  void printScop(llvm::raw_ostream &OS) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;
  static void extendScattering(Scop &S, unsigned NewDimensions);
};
}

char PlutoOptimizer::ID = 0;

static int getSingleMap(__isl_take isl_map *map, void *user) {
  isl_map **singleMap = (isl_map **)user;
  *singleMap = map;

  return 0;
}

void PlutoOptimizer::extendScattering(Scop &S, unsigned NewDimensions) {
  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    unsigned OldDimensions = Stmt->getNumScattering();
    isl_space *Space;
    isl_map *Map, *New;

    Space = isl_space_alloc(Stmt->getIslCtx(), 0, OldDimensions, NewDimensions);
    Map = isl_map_universe(Space);

    for (unsigned i = 0; i < OldDimensions; i++)
      Map = isl_map_equate(Map, isl_dim_in, i, isl_dim_out, i);

    for (unsigned i = OldDimensions; i < NewDimensions; i++)
      Map = isl_map_fix_si(Map, isl_dim_out, i, 0);

    Map = isl_map_align_params(Map, S.getParamSpace());
    New = isl_map_apply_range(Stmt->getScattering(), Map);
    Stmt->setScattering(New);
  }
}

bool PlutoOptimizer::runOnScop(Scop &S) {
  isl_union_set *Domain;
  isl_union_map *Deps, *ToPlutoNames, *Schedule;
  PlutoOptions *Options;

  Dependences *D = &getAnalysis<Dependences>();

  int DependencesKinds =
      Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;

  Deps = D->getDependences(DependencesKinds);
  Domain = S.getDomains();
  ToPlutoNames = isl_union_map_empty(S.getParamSpace());

  int counter = 0;
  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    std::string Name = "S_" + convertInt(counter);
    isl_map *Identity = isl_map_identity(isl_space_map_from_domain_and_range(
        Stmt->getDomainSpace(), Stmt->getDomainSpace()));
    Identity = isl_map_set_tuple_name(Identity, isl_dim_out, Name.c_str());
    ToPlutoNames = isl_union_map_add_map(ToPlutoNames, Identity);
    counter++;
  }

  Deps = isl_union_map_apply_domain(Deps, isl_union_map_copy(ToPlutoNames));
  Deps = isl_union_map_apply_range(Deps, isl_union_map_copy(ToPlutoNames));
  Domain = isl_union_set_apply(Domain, isl_union_map_copy(ToPlutoNames));

  Options = pluto_options_alloc();
  Options->fuse = 0;
  Options->tile = EnableTiling;

  DEBUG(dbgs() << "Domain: " << stringFromIslObj(Domain) << "\n";
        dbgs() << "Dependences: " << stringFromIslObj(Deps) << "\n";);
  Schedule = pluto_schedule(Domain, Deps, Options);
  pluto_options_free(Options);

  isl_union_set_free(Domain);
  isl_union_map_free(Deps);

  if (!Schedule)
    return false;

  Schedule =
      isl_union_map_apply_domain(Schedule, isl_union_map_reverse(ToPlutoNames));

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    isl_set *Domain = Stmt->getDomain();
    isl_union_map *StmtBand;
    StmtBand = isl_union_map_intersect_domain(isl_union_map_copy(Schedule),
                                              isl_union_set_from_set(Domain));
    isl_map *StmtSchedule;
    isl_union_map_foreach_map(StmtBand, getSingleMap, &StmtSchedule);
    Stmt->setScattering(StmtSchedule);
    isl_union_map_free(StmtBand);
  }

  isl_union_map_free(Schedule);

  unsigned MaxScatDims = 0;

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI)
    MaxScatDims = std::max((*SI)->getNumScattering(), MaxScatDims);

  extendScattering(S, MaxScatDims);
  return false;
}

void PlutoOptimizer::printScop(raw_ostream &OS) const {}

void PlutoOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}

Pass *polly::createPlutoOptimizerPass() { return new PlutoOptimizer(); }

INITIALIZE_PASS_BEGIN(PlutoOptimizer, "polly-opt-pluto",
                      "Polly - Optimize schedule of SCoP (Pluto)", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(Dependences);
INITIALIZE_PASS_DEPENDENCY(ScopInfo);
INITIALIZE_PASS_END(PlutoOptimizer, "polly-opt-pluto",
                    "Polly - Optimize schedule of SCoP (Pluto)", false, false)

#endif // PLUTO_FOUND
