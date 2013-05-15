//===-- ScopLibExporter.cpp  - Export Scops with scoplib   ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Export the Scops build by ScopInfo pass to text file.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"

#ifdef SCOPLIB_FOUND

#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/ScopLib.h"

#include "llvm/Assembly/Writer.h"

#include "stdio.h"
#include "isl/set.h"
#include "isl/constraint.h"

using namespace llvm;
using namespace polly;

namespace {
static cl::opt<std::string>
ExportDir("polly-export-scoplib-dir",
          cl::desc("The directory to export the .scoplib files to."),
          cl::Hidden, cl::value_desc("Directory path"), cl::ValueRequired,
          cl::init("."), cl::cat(PollyCategory));

class ScopLibExporter : public ScopPass {
  Scop *S;

  std::string getFileName(Scop *S) const;

public:
  static char ID;
  explicit ScopLibExporter() : ScopPass(ID) {}

  virtual bool runOnScop(Scop &scop);
  void getAnalysisUsage(AnalysisUsage &AU) const;
};

}

char ScopLibExporter::ID = 0;

std::string ScopLibExporter::getFileName(Scop *S) const {
  std::string FunctionName = S->getRegion().getEntry()->getParent()->getName();
  std::string FileName = FunctionName + "___" + S->getNameStr() + ".scoplib";
  return FileName;
}

bool ScopLibExporter::runOnScop(Scop &scop) {
  S = &scop;
  Region *R = &S->getRegion();

  std::string FileName = ExportDir + "/" + getFileName(S);
  FILE *F = fopen(FileName.c_str(), "w");

  if (!F) {
    errs() << "Cannot open file: " << FileName << "\n";
    errs() << "Skipping export.\n";
    return false;
  }

  ScopLib scoplib(S);
  scoplib.print(F);
  fclose(F);

  std::string FunctionName = R->getEntry()->getParent()->getName();
  errs() << "Writing Scop '" << R->getNameStr() << "' in function '"
         << FunctionName << "' to '" << FileName << "'.\n";

  return false;
}

void ScopLibExporter::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
}

static RegisterPass<ScopLibExporter>
A("polly-export-scoplib", "Polly - Export Scops with ScopLib library"
                          " (Writes a .scoplib file for each Scop)");

Pass *polly::createScopLibExporterPass() { return new ScopLibExporter(); }

#endif
