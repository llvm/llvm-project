//===- Pocc.cpp - Pocc interface ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pocc[1] interface.
//
// Pocc, the polyhedral compilation collection is a collection of polyhedral
// tools. It is used as an optimizer in polly
//
// [1] http://www-roc.inria.fr/~pouchet/software/pocc/
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"

#ifdef SCOPLIB_FOUND
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/Dependences.h"
#include "polly/Options.h"
#include "polly/ScheduleOptimizer.h"
#include "polly/ScopInfo.h"

#define DEBUG_TYPE "polly-opt-pocc"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"
#include "llvm/ADT/OwningPtr.h"

#include "polly/ScopLib.h"

#include "isl/space.h"
#include "isl/map.h"
#include "isl/constraint.h"

using namespace llvm;
using namespace polly;

static cl::opt<std::string> PlutoFuse("pluto-fuse", cl::desc(""), cl::Hidden,
                                      cl::value_desc("Set fuse mode of Pluto"),
                                      cl::init("maxfuse"),
                                      cl::cat(PollyCategory));

namespace {

class Pocc : public ScopPass {
  sys::Path plutoStderr;
  sys::Path plutoStdout;
  std::vector<const char *> arguments;

public:
  static char ID;
  explicit Pocc() : ScopPass(ID) {}

  std::string getFileName(Region *R) const;
  virtual bool runOnScop(Scop &S);
  void printScop(llvm::raw_ostream &OS) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;

private:
  bool runTransform(Scop &S);
};

}

char Pocc::ID = 0;
bool Pocc::runTransform(Scop &S) {
  Dependences *D = &getAnalysis<Dependences>();

  // Create the scop file.
  sys::Path tempDir = sys::Path::GetTemporaryDirectory();
  sys::Path scopFile = tempDir;
  scopFile.appendComponent("polly.scop");
  scopFile.createFileOnDisk();

  FILE *F = fopen(scopFile.c_str(), "w");

  arguments.clear();

  if (!F) {
    errs() << "Cannot open file: " << tempDir.c_str() << "\n";
    errs() << "Skipping export.\n";
    return false;
  }

  ScopLib scoplib(&S);
  scoplib.print(F);
  fclose(F);

  // Execute pocc
  sys::Program program;

  sys::Path pocc = sys::Program::FindProgramByName("pocc");

  arguments.push_back("pocc");
  arguments.push_back("--read-scop");
  arguments.push_back(scopFile.c_str());
  arguments.push_back("--pluto-tile-scat");
  arguments.push_back("--candl-dep-isl-simp");
  arguments.push_back("--cloogify-scheds");
  arguments.push_back("--output-scop");
  arguments.push_back("--pluto");
  arguments.push_back("--pluto-bounds");
  arguments.push_back("10");
  arguments.push_back("--pluto-fuse");

  arguments.push_back(PlutoFuse.c_str());

  if (!DisablePollyTiling)
    arguments.push_back("--pluto-tile");

  if (PollyVectorizerChoice != VECTORIZER_NONE)
    arguments.push_back("--pluto-prevector");

  arguments.push_back(0);

  plutoStdout = tempDir;
  plutoStdout.appendComponent("pluto.stdout");
  plutoStderr = tempDir;
  plutoStderr.appendComponent("pluto.stderr");

  std::vector<sys::Path *> redirect;
  redirect.push_back(0);
  redirect.push_back(&plutoStdout);
  redirect.push_back(&plutoStderr);

  program.ExecuteAndWait(pocc, &arguments[0], 0,
                         (sys::Path const **)&redirect[0]);

  // Read the created scop file
  sys::Path newScopFile = tempDir;
  newScopFile.appendComponent("polly.pocc.c.scop");

  FILE *poccFile = fopen(newScopFile.c_str(), "r");
  ScopLib newScoplib(&S, poccFile, D);

  if (!newScoplib.updateScattering()) {
    errs() << "Failure when calculating the optimization with "
              "the following command: ";
    for (std::vector<const char *>::const_iterator AI = arguments.begin(),
                                                   AE = arguments.end();
         AI != AE; ++AI)
      if (*AI)
        errs() << " " << *AI;
    errs() << "\n";
    return false;
  } else
    fclose(poccFile);

  if (PollyVectorizerChoice == VECTORIZER_NONE)
    return false;

  // Find the innermost dimension that is not a constant dimension. This
  // dimension will be vectorized.
  unsigned scatterDims = S.getScatterDim();
  int lastLoop = scatterDims - 1;

  while (lastLoop) {
    bool isSingleValued = true;

    for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
      isl_map *scat = (*SI)->getScattering();
      isl_map *projected = isl_map_project_out(scat, isl_dim_out, lastLoop,
                                               scatterDims - lastLoop);

      if (!isl_map_is_bijective(projected)) {
        isSingleValued = false;
        break;
      }
    }

    if (!isSingleValued)
      break;

    lastLoop--;
  }

  // Strip mine the innermost loop.
  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    isl_map *scat = (*SI)->getScattering();
    int scatDims = (*SI)->getNumScattering();
    isl_space *Space = isl_space_alloc(S.getIslCtx(), S.getNumParams(),
                                       scatDims, scatDims + 1);
    isl_basic_map *map = isl_basic_map_universe(isl_space_copy(Space));
    isl_local_space *LSpace = isl_local_space_from_space(Space);

    for (int i = 0; i <= lastLoop - 1; i++) {
      isl_constraint *c = isl_equality_alloc(isl_local_space_copy(LSpace));

      isl_constraint_set_coefficient_si(c, isl_dim_in, i, 1);
      isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1);

      map = isl_basic_map_add_constraint(map, c);
    }

    for (int i = lastLoop; i < scatDims; i++) {
      isl_constraint *c = isl_equality_alloc(isl_local_space_copy(LSpace));

      isl_constraint_set_coefficient_si(c, isl_dim_in, i, 1);
      isl_constraint_set_coefficient_si(c, isl_dim_out, i + 1, -1);

      map = isl_basic_map_add_constraint(map, c);
    }

    isl_constraint *c;

    int vectorWidth = 4;
    c = isl_inequality_alloc(isl_local_space_copy(LSpace));
    isl_constraint_set_coefficient_si(c, isl_dim_out, lastLoop, -vectorWidth);
    isl_constraint_set_coefficient_si(c, isl_dim_out, lastLoop + 1, 1);
    map = isl_basic_map_add_constraint(map, c);

    c = isl_inequality_alloc(LSpace);
    isl_constraint_set_coefficient_si(c, isl_dim_out, lastLoop, vectorWidth);
    isl_constraint_set_coefficient_si(c, isl_dim_out, lastLoop + 1, -1);
    isl_constraint_set_constant_si(c, vectorWidth - 1);
    map = isl_basic_map_add_constraint(map, c);

    isl_map *transform = isl_map_from_basic_map(map);
    transform = isl_map_set_tuple_name(transform, isl_dim_out, "scattering");
    transform = isl_map_set_tuple_name(transform, isl_dim_in, "scattering");

    scat = isl_map_apply_range(scat, isl_map_copy(transform));
    (*SI)->setScattering(scat);
  }

  return false;
}
bool Pocc::runOnScop(Scop &S) {
  bool Result = runTransform(S);
  DEBUG(printScop(dbgs()));

  return Result;
}

void Pocc::printScop(raw_ostream &OS) const {
  OwningPtr<MemoryBuffer> stdoutBuffer;
  OwningPtr<MemoryBuffer> stderrBuffer;

  OS << "Command line: ";

  for (std::vector<const char *>::const_iterator AI = arguments.begin(),
                                                 AE = arguments.end();
       AI != AE; ++AI)
    if (*AI)
      OS << " " << *AI;

  OS << "\n";

  if (error_code ec = MemoryBuffer::getFile(plutoStdout.c_str(), stdoutBuffer))
    OS << "Could not open pocc stdout file: " + ec.message() << "\n";
  else {
    OS << "pocc stdout: " << stdoutBuffer->getBufferIdentifier() << "\n";
    OS << stdoutBuffer->getBuffer() << "\n";
  }

  if (error_code ec = MemoryBuffer::getFile(plutoStderr.c_str(), stderrBuffer))
    OS << "Could not open pocc stderr file: " + ec.message() << "\n";
  else {
    OS << "pocc stderr: " << plutoStderr.c_str() << "\n";
    OS << stderrBuffer->getBuffer() << "\n";
  }
}

void Pocc::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}

Pass *polly::createPoccPass() { return new Pocc(); }

INITIALIZE_PASS_BEGIN(Pocc, "polly-opt-pocc",
                      "Polly - Optimize the scop using pocc", false, false);
INITIALIZE_PASS_DEPENDENCY(Dependences);
INITIALIZE_PASS_DEPENDENCY(ScopInfo);
INITIALIZE_PASS_END(Pocc, "polly-opt-pocc",
                    "Polly - Optimize the scop using pocc", false, false)
#endif /* SCOPLIB_FOUND */
