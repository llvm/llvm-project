//===- ProfileInfoLoaderPass.cpp - LLVM Pass to load profile info ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a concrete implementation of profiling information that
// loads the information from a profile dump file.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/InstrTypes.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Analysis/ProfileInfoLoader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"
using namespace llvm;

static cl::opt<std::string>
ProfileInfoFilename("profile-info-file", cl::init("llvmprof.out"),
                    cl::value_desc("filename"),
                    cl::desc("Profile file loaded by -profile-loader"));

namespace {
  class VISIBILITY_HIDDEN LoaderPass : public ModulePass, public ProfileInfo {
    std::string Filename;
  public:
    static char ID; // Class identification, replacement for typeinfo
    explicit LoaderPass(const std::string &filename = "")
      : ModulePass(&ID), Filename(filename) {
      if (filename.empty()) Filename = ProfileInfoFilename;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    virtual const char *getPassName() const {
      return "Profiling information loader";
    }

    /// run - Load the profile information from the specified file.
    virtual bool runOnModule(Module &M);
  };
}  // End of anonymous namespace

char LoaderPass::ID = 0;
static RegisterPass<LoaderPass>
X("profile-loader", "Load profile information from llvmprof.out", false, true);

static RegisterAnalysisGroup<ProfileInfo> Y(X);

ModulePass *llvm::createProfileLoaderPass() { return new LoaderPass(); }

/// createProfileLoaderPass - This function returns a Pass that loads the
/// profiling information for the module from the specified filename, making it
/// available to the optimizers.
Pass *llvm::createProfileLoaderPass(const std::string &Filename) {
  return new LoaderPass(Filename);
}

bool LoaderPass::runOnModule(Module &M) {
  ProfileInfoLoader PIL("profile-loader", Filename, M);
  EdgeCounts.clear();
  bool PrintedWarning = false;

  std::vector<std::pair<ProfileInfoLoader::Edge, unsigned> > ECs;
  PIL.getEdgeCounts(ECs);
  for (unsigned i = 0, e = ECs.size(); i != e; ++i) {
    BasicBlock *BB = ECs[i].first.first;
    unsigned SuccNum = ECs[i].first.second;
    TerminatorInst *TI = BB->getTerminator();
    if (SuccNum >= TI->getNumSuccessors()) {
      if (!PrintedWarning) {
        cerr << "WARNING: profile information is inconsistent with "
             << "the current program!\n";
        PrintedWarning = true;
      }
    } else {
      EdgeCounts[std::make_pair(BB, TI->getSuccessor(SuccNum))]+= ECs[i].second;
    }
  }

  return false;
}
