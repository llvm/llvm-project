//===- ProfileDataLoaderPass.cpp - Set branch weight metadata from prof ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass loads profiling data from a dump file and sets branch weight
// metadata.
//
// TODO: Replace all "profile-metadata-loader" strings with "profile-loader"
// once ProfileInfo etc. has been removed.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "profile-metadata-loader"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/BasicBlock.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#include "llvm/LLVMContext.h"
#include "llvm/MDBuilder.h"
#include "llvm/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileDataLoader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumEdgesRead, "The # of edges read.");
STATISTIC(NumTermsAnnotated, "The # of terminator instructions annotated.");

static cl::opt<std::string>
ProfileMetadataFilename("profile-file", cl::init("llvmprof.out"),
                  cl::value_desc("filename"),
                  cl::desc("Profile file loaded by -profile-metadata-loader"));

namespace {
  /// This pass loads profiling data from a dump file and sets branch weight
  /// metadata.
  class ProfileMetadataLoaderPass : public ModulePass {
    std::string Filename;
  public:
    static char ID; // Class identification, replacement for typeinfo
    explicit ProfileMetadataLoaderPass(const std::string &filename = "")
        : ModulePass(ID), Filename(filename) {
      initializeProfileMetadataLoaderPassPass(*PassRegistry::getPassRegistry());
      if (filename.empty()) Filename = ProfileMetadataFilename;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    virtual const char *getPassName() const {
      return "Profile loader";
    }

    virtual void readEdge(unsigned, ProfileData&, ProfileData::Edge,
                          ArrayRef<unsigned>);
    virtual unsigned matchEdges(Module&, ProfileData&, ArrayRef<unsigned>);
    virtual void setBranchWeightMetadata(Module&, ProfileData&);

    virtual bool runOnModule(Module &M);
  };
}  // End of anonymous namespace

char ProfileMetadataLoaderPass::ID = 0;
INITIALIZE_PASS_BEGIN(ProfileMetadataLoaderPass, "profile-metadata-loader",
              "Load profile information from llvmprof.out", false, true)
INITIALIZE_PASS_END(ProfileMetadataLoaderPass, "profile-metadata-loader",
              "Load profile information from llvmprof.out", false, true)

char &llvm::ProfileMetadataLoaderPassID = ProfileMetadataLoaderPass::ID;

/// createProfileMetadataLoaderPass - This function returns a Pass that loads
/// the profiling information for the module from the specified filename,
/// making it available to the optimizers.
ModulePass *llvm::createProfileMetadataLoaderPass() { 
    return new ProfileMetadataLoaderPass();
}
ModulePass *llvm::createProfileMetadataLoaderPass(const std::string &Filename) {
  return new ProfileMetadataLoaderPass(Filename);
}

/// readEdge - Take the value from a profile counter and assign it to an edge.
void ProfileMetadataLoaderPass::readEdge(unsigned ReadCount,
                                         ProfileData &PB, ProfileData::Edge e,
                                         ArrayRef<unsigned> Counters) {
  if (ReadCount >= Counters.size()) return;

  unsigned weight = Counters[ReadCount];
  assert(weight != ProfileDataLoader::Uncounted);
  PB.addEdgeWeight(e, weight);

  DEBUG(dbgs() << "-- Read Edge Counter for " << e
               << " (# "<< (ReadCount) << "): "
               << PB.getEdgeWeight(e) << "\n");
}

/// matchEdges - Link every profile counter with an edge.
unsigned ProfileMetadataLoaderPass::matchEdges(Module &M, ProfileData &PB,
                                               ArrayRef<unsigned> Counters) {
  if (Counters.size() == 0) return 0;

  unsigned ReadCount = 0;

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    DEBUG(dbgs() << "Loading edges in '" << F->getName() << "'\n");
    readEdge(ReadCount++, PB, PB.getEdge(0, &F->getEntryBlock()), Counters);
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      TerminatorInst *TI = BB->getTerminator();
      for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
        readEdge(ReadCount++, PB, PB.getEdge(BB,TI->getSuccessor(s)),
                 Counters);
      }
    }
  }

  return ReadCount;
}

/// setBranchWeightMetadata - Translate the counter values associated with each
/// edge into branch weights for each conditional branch (a branch with 2 or
/// more desinations).
void ProfileMetadataLoaderPass::setBranchWeightMetadata(Module &M,
                                                        ProfileData &PB) {
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    DEBUG(dbgs() << "Setting branch metadata in '" << F->getName() << "'\n");

    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      TerminatorInst *TI = BB->getTerminator();
      unsigned NumSuccessors = TI->getNumSuccessors();

      // If there is only one successor then we can not set a branch
      // probability as the target is certain.
      if (NumSuccessors < 2) continue;

      // Load the weights of all edges leading from this terminator.
      DEBUG(dbgs() << "-- Terminator with " << NumSuccessors
                   << " successors:\n");
      SmallVector<uint32_t, 4> Weights(NumSuccessors);
      for (unsigned s = 0 ; s < NumSuccessors ; ++s) {
          ProfileData::Edge edge = PB.getEdge(BB, TI->getSuccessor(s));
          Weights[s] = (uint32_t)PB.getEdgeWeight(edge);
          DEBUG(dbgs() << "---- Edge '" << edge << "' has weight "
                       << Weights[s] << "\n");
      }

      // Set branch weight metadata.  This will set branch probabilities of
      // 100%/0% if that is true of the dynamic execution.
      // BranchProbabilityInfo can account for this when it loads this metadata
      // (it gives the unexectuted branch a weight of 1 for the purposes of
      // probability calculations).
      MDBuilder MDB(TI->getContext());
      MDNode *Node = MDB.createBranchWeights(Weights);
      TI->setMetadata(LLVMContext::MD_prof, Node);
      NumTermsAnnotated++;
    }
  }
}

bool ProfileMetadataLoaderPass::runOnModule(Module &M) {
  ProfileDataLoader PDL("profile-data-loader", Filename);
  ProfileData PB;

  ArrayRef<unsigned> Counters = PDL.getRawEdgeCounts();

  unsigned ReadCount = matchEdges(M, PB, Counters);

  if (ReadCount != Counters.size()) {
    errs() << "WARNING: profile information is inconsistent with "
           << "the current program!\n";
  }
  NumEdgesRead = ReadCount;

  setBranchWeightMetadata(M, PB);

  return ReadCount > 0;
}
