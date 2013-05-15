//===-- ScopLibImporter.cpp  - Import Scops with scoplib. -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Import modified .scop files into Polly. This allows to change the schedule of
// statements.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"

#ifdef SCOPLIB_FOUND

#include "polly/Dependences.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopLib.h"

#include "llvm/Assembly/Writer.h"

#define SCOPLIB_INT_T_IS_MP
#include "scoplib/scop.h"

#include "isl/set.h"
#include "isl/constraint.h"

using namespace llvm;
using namespace polly;

namespace {
static cl::opt<std::string>
ImportDir("polly-import-scoplib-dir",
          cl::desc("The directory to import the .scoplib files from."),
          cl::Hidden, cl::value_desc("Directory path"), cl::ValueRequired,
          cl::init("."), cl::cat(PollyCategory));

static cl::opt<std::string>
ImportPostfix("polly-import-scoplib-postfix",
              cl::desc("Postfix to append to the import .scoplib files."),
              cl::Hidden, cl::value_desc("File postfix"), cl::ValueRequired,
              cl::init(""), cl::cat(PollyCategory));

struct ScopLibImporter : public RegionPass {
  static char ID;
  Scop *S;
  Dependences *D;
  explicit ScopLibImporter() : RegionPass(ID) {}

  bool updateScattering(Scop *S, scoplib_scop_p OScop);
  std::string getFileName(Scop *S) const;
  virtual bool runOnRegion(Region *R, RGPassManager &RGM);
  virtual void print(raw_ostream &OS, const Module *) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;
};
}

char ScopLibImporter::ID = 0;

namespace {
std::string ScopLibImporter::getFileName(Scop *S) const {
  std::string FunctionName = S->getRegion().getEntry()->getParent()->getName();
  std::string FileName = FunctionName + "___" + S->getNameStr() + ".scoplib";
  return FileName;
}

void ScopLibImporter::print(raw_ostream &OS, const Module *) const {}

bool ScopLibImporter::runOnRegion(Region *R, RGPassManager &RGM) {
  S = getAnalysis<ScopInfo>().getScop();
  D = &getAnalysis<Dependences>();

  if (!S)
    return false;

  std::string FileName = ImportDir + "/" + getFileName(S) + ImportPostfix;
  FILE *F = fopen(FileName.c_str(), "r");

  if (!F) {
    errs() << "Cannot open file: " << FileName << "\n";
    errs() << "Skipping import.\n";
    return false;
  }

  std::string FunctionName = R->getEntry()->getParent()->getName();
  errs() << "Reading Scop '" << R->getNameStr() << "' in function '"
         << FunctionName << "' from '" << FileName << "'.\n";

  ScopLib scoplib(S, F, D);
  bool UpdateSuccessfull = scoplib.updateScattering();
  fclose(F);

  if (!UpdateSuccessfull) {
    errs() << "Update failed"
           << "\n";
  }

  return false;
}

void ScopLibImporter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<ScopInfo>();
  AU.addRequired<Dependences>();
}

}

static RegisterPass<ScopLibImporter>
A("polly-import-scoplib", "Polly - Import Scops with ScopLib library"
                          " (Reads a .scoplib file for each Scop)");

Pass *polly::createScopLibImporterPass() { return new ScopLibImporter(); }

#endif
