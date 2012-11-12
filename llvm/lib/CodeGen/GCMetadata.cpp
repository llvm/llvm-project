//===-- GCMetadata.cpp - Garbage collector metadata -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GCFunctionInfo class and GCModuleInfo pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  
  class Printer : public FunctionPass {
    static char ID;
    raw_ostream &OS;
    
  public:
    explicit Printer(raw_ostream &OS) : FunctionPass(ID), OS(OS) {}

    
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnFunction(Function &F);
  };
  
  class Deleter : public FunctionPass {
    static char ID;
    
  public:
    Deleter();
    
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnFunction(Function &F);
    bool doFinalization(Module &M);
  };
  
}

INITIALIZE_PASS(GCModuleInfo, "collector-metadata",
                "Create Garbage Collector Module Metadata", false, false)

// -----------------------------------------------------------------------------

GCFunctionInfo::GCFunctionInfo(const Function &F, GCStrategy &S)
  : F(F), S(S), FrameSize(~0LL) {}

GCFunctionInfo::~GCFunctionInfo() {}

// -----------------------------------------------------------------------------

char GCModuleInfo::ID = 0;

GCModuleInfo::GCModuleInfo()
    : ImmutablePass(ID) {
  initializeGCModuleInfoPass(*PassRegistry::getPassRegistry());
}

GCModuleInfo::~GCModuleInfo() {
  clear();
}

GCStrategy *GCModuleInfo::getOrCreateStrategy(const Module *M,
                                              const std::string &Name) {
  strategy_map_type::iterator NMI = StrategyMap.find(Name);
  if (NMI != StrategyMap.end())
    return NMI->getValue();
  
  for (GCRegistry::iterator I = GCRegistry::begin(),
                            E = GCRegistry::end(); I != E; ++I) {
    if (Name == I->getName()) {
      GCStrategy *S = I->instantiate();
      S->M = M;
      S->Name = Name;
      StrategyMap.GetOrCreateValue(Name).setValue(S);
      StrategyList.push_back(S);
      return S;
    }
  }
 
  dbgs() << "unsupported GC: " << Name << "\n";
  llvm_unreachable(0);
}

GCFunctionInfo &GCModuleInfo::getFunctionInfo(const Function &F) {
  assert(!F.isDeclaration() && "Can only get GCFunctionInfo for a definition!");
  assert(F.hasGC());
  
  finfo_map_type::iterator I = FInfoMap.find(&F);
  if (I != FInfoMap.end())
    return *I->second;
  
  GCStrategy *S = getOrCreateStrategy(F.getParent(), F.getGC());
  GCFunctionInfo *GFI = S->insertFunctionInfo(F);
  FInfoMap[&F] = GFI;
  return *GFI;
}

void GCModuleInfo::clear() {
  FInfoMap.clear();
  StrategyMap.clear();
  
  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;
  StrategyList.clear();
}

// -----------------------------------------------------------------------------

char Printer::ID = 0;

FunctionPass *llvm::createGCInfoPrinter(raw_ostream &OS) {
  return new Printer(OS);
}


const char *Printer::getPassName() const {
  return "Print Garbage Collector Information";
}

void Printer::getAnalysisUsage(AnalysisUsage &AU) const {
  FunctionPass::getAnalysisUsage(AU);
  AU.setPreservesAll();
  AU.addRequired<GCModuleInfo>();
}

static const char *DescKind(GC::PointKind Kind) {
  switch (Kind) {
    case GC::Loop:     return "loop";
    case GC::Return:   return "return";
    case GC::PreCall:  return "pre-call";
    case GC::PostCall: return "post-call";
  }
  llvm_unreachable("Invalid point kind");
}

bool Printer::runOnFunction(Function &F) {
  if (F.hasGC()) return false;
  
  GCFunctionInfo *FD = &getAnalysis<GCModuleInfo>().getFunctionInfo(F);
  
  OS << "GC roots for " << FD->getFunction().getName() << ":\n";
  for (GCFunctionInfo::roots_iterator RI = FD->roots_begin(),
                                      RE = FD->roots_end(); RI != RE; ++RI)
    OS << "\t" << RI->Num << "\t" << RI->StackOffset << "[sp]\n";
  
  OS << "GC safe points for " << FD->getFunction().getName() << ":\n";
  for (GCFunctionInfo::iterator PI = FD->begin(),
                                PE = FD->end(); PI != PE; ++PI) {
    
    OS << "\t" << PI->Label->getName() << ": "
       << DescKind(PI->Kind) << ", live = {";
    
    for (GCFunctionInfo::live_iterator RI = FD->live_begin(PI),
                                       RE = FD->live_end(PI);;) {
      OS << " " << RI->Num;
      if (++RI == RE)
        break;
      OS << ",";
    }
    
    OS << " }\n";
  }
  
  return false;
}

// -----------------------------------------------------------------------------

char Deleter::ID = 0;

FunctionPass *llvm::createGCInfoDeleter() {
  return new Deleter();
}

Deleter::Deleter() : FunctionPass(ID) {}

const char *Deleter::getPassName() const {
  return "Delete Garbage Collector Information";
}

void Deleter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<GCModuleInfo>();
}

bool Deleter::runOnFunction(Function &MF) {
  return false;
}

bool Deleter::doFinalization(Module &M) {
  GCModuleInfo *GMI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(GMI && "Deleter didn't require GCModuleInfo?!");
  GMI->clear();
  return false;
}
