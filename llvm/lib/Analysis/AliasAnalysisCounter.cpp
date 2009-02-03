//===- AliasAnalysisCounter.cpp - Alias Analysis Query Counter ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass which can be used to count how many alias queries
// are being made and how the alias analysis implementation being used responds.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"
using namespace llvm;

static cl::opt<bool>
PrintAll("count-aa-print-all-queries", cl::ReallyHidden);
static cl::opt<bool>
PrintAllFailures("count-aa-print-all-failed-queries", cl::ReallyHidden);

namespace {
  class VISIBILITY_HIDDEN AliasAnalysisCounter 
      : public ModulePass, public AliasAnalysis {
    unsigned No, May, Must;
    unsigned NoMR, JustRef, JustMod, MR;
    const char *Name;
    Module *M;
  public:
    static char ID; // Class identification, replacement for typeinfo
    AliasAnalysisCounter() : ModulePass(&ID) {
      No = May = Must = 0;
      NoMR = JustRef = JustMod = MR = 0;
    }

    void printLine(const char *Desc, unsigned Val, unsigned Sum) {
      cerr <<  "  " << Val << " " << Desc << " responses ("
           << Val*100/Sum << "%)\n";
    }
    ~AliasAnalysisCounter() {
      unsigned AASum = No+May+Must;
      unsigned MRSum = NoMR+JustRef+JustMod+MR;
      if (AASum + MRSum) { // Print a report if any counted queries occurred...
        cerr << "\n===== Alias Analysis Counter Report =====\n"
             << "  Analysis counted: " << Name << "\n"
             << "  " << AASum << " Total Alias Queries Performed\n";
        if (AASum) {
          printLine("no alias",     No, AASum);
          printLine("may alias",   May, AASum);
          printLine("must alias", Must, AASum);
          cerr << "  Alias Analysis Counter Summary: " << No*100/AASum << "%/"
               << May*100/AASum << "%/" << Must*100/AASum<<"%\n\n";
        }

        cerr << "  " << MRSum    << " Total Mod/Ref Queries Performed\n";
        if (MRSum) {
          printLine("no mod/ref",    NoMR, MRSum);
          printLine("ref",        JustRef, MRSum);
          printLine("mod",        JustMod, MRSum);
          printLine("mod/ref",         MR, MRSum);
          cerr << "  Mod/Ref Analysis Counter Summary: " <<NoMR*100/MRSum<< "%/"
               << JustRef*100/MRSum << "%/" << JustMod*100/MRSum << "%/"
               << MR*100/MRSum <<"%\n\n";
        }
      }
    }

    bool runOnModule(Module &M) {
      this->M = &M;
      InitializeAliasAnalysis(this);
      Name = dynamic_cast<Pass*>(&getAnalysis<AliasAnalysis>())->getPassName();
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.setPreservesAll();
    }

    // FIXME: We could count these too...
    bool pointsToConstantMemory(const Value *P) {
      return getAnalysis<AliasAnalysis>().pointsToConstantMemory(P);
    }
    bool doesNotAccessMemory(CallSite CS) {
      return getAnalysis<AliasAnalysis>().doesNotAccessMemory(CS);
    }
    bool doesNotAccessMemory(Function *F) {
      return getAnalysis<AliasAnalysis>().doesNotAccessMemory(F);
    }
    bool onlyReadsMemory(CallSite CS) {
      return getAnalysis<AliasAnalysis>().onlyReadsMemory(CS);
    }
    bool onlyReadsMemory(Function *F) {
      return getAnalysis<AliasAnalysis>().onlyReadsMemory(F);
    }


    // Forwarding functions: just delegate to a real AA implementation, counting
    // the number of responses...
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);

    ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);
    ModRefResult getModRefInfo(CallSite CS1, CallSite CS2) {
      return AliasAnalysis::getModRefInfo(CS1,CS2);
    }
  };
}

char AliasAnalysisCounter::ID = 0;
static RegisterPass<AliasAnalysisCounter>
X("count-aa", "Count Alias Analysis Query Responses", false, true);
static RegisterAnalysisGroup<AliasAnalysis> Y(X);

ModulePass *llvm::createAliasAnalysisCounterPass() {
  return new AliasAnalysisCounter();
}

AliasAnalysis::AliasResult
AliasAnalysisCounter::alias(const Value *V1, unsigned V1Size,
                            const Value *V2, unsigned V2Size) {
  AliasResult R = getAnalysis<AliasAnalysis>().alias(V1, V1Size, V2, V2Size);

  const char *AliasString;
  switch (R) {
  default: assert(0 && "Unknown alias type!");
  case NoAlias:   No++;   AliasString = "No alias"; break;
  case MayAlias:  May++;  AliasString = "May alias"; break;
  case MustAlias: Must++; AliasString = "Must alias"; break;
  }

  if (PrintAll || (PrintAllFailures && R == MayAlias)) {
    cerr << AliasString << ":\t";
    cerr << "[" << V1Size << "B] ";
    WriteAsOperand(*cerr.stream(), V1, true, M);
    cerr << ", ";
    cerr << "[" << V2Size << "B] ";
    WriteAsOperand(*cerr.stream(), V2, true, M);
    cerr << "\n";
  }

  return R;
}

AliasAnalysis::ModRefResult
AliasAnalysisCounter::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  ModRefResult R = getAnalysis<AliasAnalysis>().getModRefInfo(CS, P, Size);

  const char *MRString;
  switch (R) {
  default:       assert(0 && "Unknown mod/ref type!");
  case NoModRef: NoMR++;     MRString = "NoModRef"; break;
  case Ref:      JustRef++;  MRString = "JustRef"; break;
  case Mod:      JustMod++;  MRString = "JustMod"; break;
  case ModRef:   MR++;       MRString = "ModRef"; break;
  }

  if (PrintAll || (PrintAllFailures && R == ModRef)) {
    cerr << MRString << ":  Ptr: ";
    cerr << "[" << Size << "B] ";
    WriteAsOperand(*cerr.stream(), P, true, M);
    cerr << "\t<->" << *CS.getInstruction();
  }
  return R;
}
