//===- ProfileVerifierPass.cpp - LLVM Pass to estimate profile info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that checks profiling information for 
// plausibility.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "profile-verifier"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Debug.h"
#include <set>
using namespace llvm;

static cl::opt<bool,false>
ProfileVerifierDisableAssertions("profile-verifier-noassert",
     cl::desc("Disable assertions"));

namespace {
  template<class FType, class BType>
  class ProfileVerifierPassT : public FunctionPass {

    struct DetailedBlockInfo {
      const BType *BB;
      double      BBWeight;
      double      inWeight;
      int         inCount;
      double      outWeight;
      int         outCount;
    };

    ProfileInfoT<FType, BType> *PI;
    std::set<const BType*> BBisVisited;
    std::set<const FType*>   FisVisited;
    bool DisableAssertions;

    // When debugging is enabled, the verifier prints a whole slew of debug
    // information, otherwise its just the assert. These are all the helper
    // functions.
    bool PrintedDebugTree;
    std::set<const BType*> BBisPrinted;
    void debugEntry(DetailedBlockInfo*);
    void printDebugInfo(const BType *BB);

  public:
    static char ID; // Class identification, replacement for typeinfo

    explicit ProfileVerifierPassT () : FunctionPass(ID) {
      initializeProfileVerifierPassPass(*PassRegistry::getPassRegistry());
      DisableAssertions = ProfileVerifierDisableAssertions;
    }
    explicit ProfileVerifierPassT (bool da) : FunctionPass(ID), 
                                              DisableAssertions(da) {
      initializeProfileVerifierPassPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<ProfileInfoT<FType, BType> >();
    }

    const char *getPassName() const {
      return "Profiling information verifier";
    }

    /// run - Verify the profile information.
    bool runOnFunction(FType &F);
    void recurseBasicBlock(const BType*);

    bool   exitReachable(const FType*);
    double ReadOrAssert(typename ProfileInfoT<FType, BType>::Edge);
    void   CheckValue(bool, const char*, DetailedBlockInfo*);
  };

  typedef ProfileVerifierPassT<Function, BasicBlock> ProfileVerifierPass;

  template<class FType, class BType>
  void ProfileVerifierPassT<FType, BType>::printDebugInfo(const BType *BB) {

    if (BBisPrinted.find(BB) != BBisPrinted.end()) return;

    double BBWeight = PI->getExecutionCount(BB);
    if (BBWeight == ProfileInfoT<FType, BType>::MissingValue) { BBWeight = 0; }
    double inWeight = 0;
    int inCount = 0;
    std::set<const BType*> ProcessedPreds;
    for (const_pred_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
         bbi != bbe; ++bbi ) {
      if (ProcessedPreds.insert(*bbi).second) {
        typename ProfileInfoT<FType, BType>::Edge E = PI->getEdge(*bbi,BB);
        double EdgeWeight = PI->getEdgeWeight(E);
        if (EdgeWeight == ProfileInfoT<FType, BType>::MissingValue) { EdgeWeight = 0; }
        dbgs() << "calculated in-edge " << E << ": " 
               << format("%20.20g",EdgeWeight) << "\n";
        inWeight += EdgeWeight;
        inCount++;
      }
    }
    double outWeight = 0;
    int outCount = 0;
    std::set<const BType*> ProcessedSuccs;
    for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
          bbi != bbe; ++bbi ) {
      if (ProcessedSuccs.insert(*bbi).second) {
        typename ProfileInfoT<FType, BType>::Edge E = PI->getEdge(BB,*bbi);
        double EdgeWeight = PI->getEdgeWeight(E);
        if (EdgeWeight == ProfileInfoT<FType, BType>::MissingValue) { EdgeWeight = 0; }
        dbgs() << "calculated out-edge " << E << ": " 
               << format("%20.20g",EdgeWeight) << "\n";
        outWeight += EdgeWeight;
        outCount++;
      }
    }
    dbgs() << "Block " << BB->getName()                   << " in "
           << BB->getParent()->getName()                  << ":"
           << "BBWeight="  << format("%20.20g",BBWeight)  << ","
           << "inWeight="  << format("%20.20g",inWeight)  << ","
           << "inCount="   << inCount                     << ","
           << "outWeight=" << format("%20.20g",outWeight) << ","
           << "outCount"   << outCount                    << "\n";

    // mark as visited and recurse into subnodes
    BBisPrinted.insert(BB);
    for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB); 
          bbi != bbe; ++bbi ) {
      printDebugInfo(*bbi);
    }
  }

  template<class FType, class BType>
  void ProfileVerifierPassT<FType, BType>::debugEntry (DetailedBlockInfo *DI) {
    dbgs() << "TROUBLE: Block " << DI->BB->getName()          << " in "
           << DI->BB->getParent()->getName()                  << ":"
           << "BBWeight="  << format("%20.20g",DI->BBWeight)  << ","
           << "inWeight="  << format("%20.20g",DI->inWeight)  << ","
           << "inCount="   << DI->inCount                     << ","
           << "outWeight=" << format("%20.20g",DI->outWeight) << ","
           << "outCount="  << DI->outCount                    << "\n";
    if (!PrintedDebugTree) {
      PrintedDebugTree = true;
      printDebugInfo(&(DI->BB->getParent()->getEntryBlock()));
    }
  }

  // This compares A and B for equality.
  static bool Equals(double A, double B) {
    return A == B;
  }

  // This checks if the function "exit" is reachable from an given function
  // via calls, this is necessary to check if a profile is valid despite the
  // counts not fitting exactly.
  template<class FType, class BType>
  bool ProfileVerifierPassT<FType, BType>::exitReachable(const FType *F) {
    if (!F) return false;

    if (FisVisited.count(F)) return false;

    FType *Exit = F->getParent()->getFunction("exit");
    if (Exit == F) {
      return true;
    }

    FisVisited.insert(F);
    bool exits = false;
    for (const_inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (const CallInst *CI = dyn_cast<CallInst>(&*I)) {
        FType *F = CI->getCalledFunction();
        if (F) {
          exits |= exitReachable(F);
        } else {
          // This is a call to a pointer, all bets are off...
          exits = true;
        }
        if (exits) break;
      }
    }
    return exits;
  }

  #define ASSERTMESSAGE(M) \
    { dbgs() << "ASSERT:" << (M) << "\n"; \
      if (!DisableAssertions) assert(0 && (M)); }

  template<class FType, class BType>
  double ProfileVerifierPassT<FType, BType>::ReadOrAssert(typename ProfileInfoT<FType, BType>::Edge E) {
    double EdgeWeight = PI->getEdgeWeight(E);
    if (EdgeWeight == ProfileInfoT<FType, BType>::MissingValue) {
      dbgs() << "Edge " << E << " in Function " 
             << ProfileInfoT<FType, BType>::getFunction(E)->getName() << ": ";
      ASSERTMESSAGE("Edge has missing value");
      return 0;
    } else {
      if (EdgeWeight < 0) {
        dbgs() << "Edge " << E << " in Function " 
               << ProfileInfoT<FType, BType>::getFunction(E)->getName() << ": ";
        ASSERTMESSAGE("Edge has negative value");
      }
      return EdgeWeight;
    }
  }

  template<class FType, class BType>
  void ProfileVerifierPassT<FType, BType>::CheckValue(bool Error, 
                                                      const char *Message,
                                                      DetailedBlockInfo *DI) {
    if (Error) {
      DEBUG(debugEntry(DI));
      dbgs() << "Block " << DI->BB->getName() << " in Function "
             << DI->BB->getParent()->getName() << ": ";
      ASSERTMESSAGE(Message);
    }
    return;
  }

  // This calculates the Information for a block and then recurses into the
  // successors.
  template<class FType, class BType>
  void ProfileVerifierPassT<FType, BType>::recurseBasicBlock(const BType *BB) {

    // Break the recursion by remembering all visited blocks.
    if (BBisVisited.find(BB) != BBisVisited.end()) return;

    // Use a data structure to store all the information, this can then be handed
    // to debug printers.
    DetailedBlockInfo DI;
    DI.BB = BB;
    DI.outCount = DI.inCount = 0;
    DI.inWeight = DI.outWeight = 0;

    // Read predecessors.
    std::set<const BType*> ProcessedPreds;
    const_pred_iterator bpi = pred_begin(BB), bpe = pred_end(BB);
    // If there are none, check for (0,BB) edge.
    if (bpi == bpe) {
      DI.inWeight += ReadOrAssert(PI->getEdge(0,BB));
      DI.inCount++;
    }
    for (;bpi != bpe; ++bpi) {
      if (ProcessedPreds.insert(*bpi).second) {
        DI.inWeight += ReadOrAssert(PI->getEdge(*bpi,BB));
        DI.inCount++;
      }
    }

    // Read successors.
    std::set<const BType*> ProcessedSuccs;
    succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
    // If there is an (0,BB) edge, consider it too. (This is done not only when
    // there are no successors, but every time; not every function contains
    // return blocks with no successors (think loop latch as return block)).
    double w = PI->getEdgeWeight(PI->getEdge(BB,0));
    if (w != ProfileInfoT<FType, BType>::MissingValue) {
      DI.outWeight += w;
      DI.outCount++;
    }
    for (;bbi != bbe; ++bbi) {
      if (ProcessedSuccs.insert(*bbi).second) {
        DI.outWeight += ReadOrAssert(PI->getEdge(BB,*bbi));
        DI.outCount++;
      }
    }

    // Read block weight.
    DI.BBWeight = PI->getExecutionCount(BB);
    CheckValue(DI.BBWeight == ProfileInfoT<FType, BType>::MissingValue,
               "BasicBlock has missing value", &DI);
    CheckValue(DI.BBWeight < 0,
               "BasicBlock has negative value", &DI);

    // Check if this block is a setjmp target.
    bool isSetJmpTarget = false;
    if (DI.outWeight > DI.inWeight) {
      for (typename BType::const_iterator i = BB->begin(), ie = BB->end();
           i != ie; ++i) {
        if (const CallInst *CI = dyn_cast<CallInst>(&*i)) {
          FType *F = CI->getCalledFunction();
          if (F && (F->getName() == "_setjmp")) {
            isSetJmpTarget = true; break;
          }
        }
      }
    }
    // Check if this block is eventually reaching exit.
    bool isExitReachable = false;
    if (DI.inWeight > DI.outWeight) {
      for (typename BType::const_iterator i = BB->begin(), ie = BB->end();
           i != ie; ++i) {
        if (const CallInst *CI = dyn_cast<CallInst>(&*i)) {
          FType *F = CI->getCalledFunction();
          if (F) {
            FisVisited.clear();
            isExitReachable |= exitReachable(F);
          } else {
            // This is a call to a pointer, all bets are off...
            isExitReachable = true;
          }
          if (isExitReachable) break;
        }
      }
    }

    if (DI.inCount > 0 && DI.outCount == 0) {
       // If this is a block with no successors.
      if (!isSetJmpTarget) {
        CheckValue(!Equals(DI.inWeight,DI.BBWeight), 
                   "inWeight and BBWeight do not match", &DI);
      }
    } else if (DI.inCount == 0 && DI.outCount > 0) {
      // If this is a block with no predecessors.
      if (!isExitReachable)
        CheckValue(!Equals(DI.BBWeight,DI.outWeight), 
                   "BBWeight and outWeight do not match", &DI);
    } else {
      // If this block has successors and predecessors.
      if (DI.inWeight > DI.outWeight && !isExitReachable)
        CheckValue(!Equals(DI.inWeight,DI.outWeight), 
                   "inWeight and outWeight do not match", &DI);
      if (DI.inWeight < DI.outWeight && !isSetJmpTarget)
        CheckValue(!Equals(DI.inWeight,DI.outWeight), 
                   "inWeight and outWeight do not match", &DI);
    }


    // Mark this block as visited, rescurse into successors.
    BBisVisited.insert(BB);
    for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB); 
          bbi != bbe; ++bbi ) {
      recurseBasicBlock(*bbi);
    }
  }

  template<class FType, class BType>
  bool ProfileVerifierPassT<FType, BType>::runOnFunction(FType &F) {
    PI = getAnalysisIfAvailable<ProfileInfoT<FType, BType> >();
    if (!PI)
      ASSERTMESSAGE("No ProfileInfo available");

    // Prepare global variables.
    PrintedDebugTree = false;
    BBisVisited.clear();

    // Fetch entry block and recurse into it.
    const BType *entry = &F.getEntryBlock();
    recurseBasicBlock(entry);

    if (PI->getExecutionCount(&F) != PI->getExecutionCount(entry))
      ASSERTMESSAGE("Function count and entry block count do not match");

    return false;
  }

  template<class FType, class BType>
  char ProfileVerifierPassT<FType, BType>::ID = 0;
}

INITIALIZE_PASS_BEGIN(ProfileVerifierPass, "profile-verifier",
                "Verify profiling information", false, true)
INITIALIZE_AG_DEPENDENCY(ProfileInfo)
INITIALIZE_PASS_END(ProfileVerifierPass, "profile-verifier",
                "Verify profiling information", false, true)

namespace llvm {
  FunctionPass *createProfileVerifierPass() {
    return new ProfileVerifierPass(ProfileVerifierDisableAssertions); 
  }
}

