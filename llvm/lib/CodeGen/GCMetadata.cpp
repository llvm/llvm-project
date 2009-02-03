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
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace {
  
  class VISIBILITY_HIDDEN Printer : public FunctionPass {
    static char ID;
    std::ostream &OS;
    
  public:
    explicit Printer(std::ostream &OS = *cerr);
    
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnFunction(Function &F);
  };
  
  class VISIBILITY_HIDDEN Deleter : public FunctionPass {
    static char ID;
    
  public:
    Deleter();
    
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnFunction(Function &F);
    bool doFinalization(Module &M);
  };
  
}

static RegisterPass<GCModuleInfo>
X("collector-metadata", "Create Garbage Collector Module Metadata");

// -----------------------------------------------------------------------------

GCFunctionInfo::GCFunctionInfo(const Function &F, GCStrategy &S)
  : F(F), S(S), FrameSize(~0LL) {}

GCFunctionInfo::~GCFunctionInfo() {}

// -----------------------------------------------------------------------------

char GCModuleInfo::ID = 0;

GCModuleInfo::GCModuleInfo()
  : ImmutablePass(&ID) {}

GCModuleInfo::~GCModuleInfo() {
  clear();
}

GCStrategy *GCModuleInfo::getOrCreateStrategy(const Module *M,
                                              const std::string &Name) {
  const char *Start = Name.c_str();
  
  strategy_map_type::iterator NMI =
    StrategyMap.find(Start, Start + Name.size());
  if (NMI != StrategyMap.end())
    return NMI->getValue();
  
  for (GCRegistry::iterator I = GCRegistry::begin(),
                            E = GCRegistry::end(); I != E; ++I) {
    if (strcmp(Start, I->getName()) == 0) {
      GCStrategy *S = I->instantiate();
      S->M = M;
      S->Name = Name;
      StrategyMap.GetOrCreateValue(Start, Start + Name.size()).setValue(S);
      StrategyList.push_back(S);
      return S;
    }
  }
  
  cerr << "unsupported GC: " << Name << "\n";
  abort();
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

FunctionPass *llvm::createGCInfoPrinter(std::ostream &OS) {
  return new Printer(OS);
}

Printer::Printer(std::ostream &OS)
  : FunctionPass(intptr_t(&ID)), OS(OS) {}

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
    default: assert(0 && "Unknown GC point kind");
    case GC::Loop:     return "loop";
    case GC::Return:   return "return";
    case GC::PreCall:  return "pre-call";
    case GC::PostCall: return "post-call";
  }
}

bool Printer::runOnFunction(Function &F) {
  if (!F.hasGC()) {
    GCFunctionInfo *FD = &getAnalysis<GCModuleInfo>().getFunctionInfo(F);
    
    OS << "GC roots for " << FD->getFunction().getNameStart() << ":\n";
    for (GCFunctionInfo::roots_iterator RI = FD->roots_begin(),
                                        RE = FD->roots_end(); RI != RE; ++RI)
      OS << "\t" << RI->Num << "\t" << RI->StackOffset << "[sp]\n";
    
    OS << "GC safe points for " << FD->getFunction().getNameStart() << ":\n";
    for (GCFunctionInfo::iterator PI = FD->begin(),
                                  PE = FD->end(); PI != PE; ++PI) {
      
      OS << "\tlabel " << PI->Num << ": " << DescKind(PI->Kind) << ", live = {";
      
      for (GCFunctionInfo::live_iterator RI = FD->live_begin(PI),
                                         RE = FD->live_end(PI);;) {
        OS << " " << RI->Num;
        if (++RI == RE)
          break;
        OS << ",";
      }
      
      OS << " }\n";
    }
  }
  
  return false;
}

// -----------------------------------------------------------------------------

char Deleter::ID = 0;

FunctionPass *llvm::createGCInfoDeleter() {
  return new Deleter();
}

Deleter::Deleter() : FunctionPass(intptr_t(&ID)) {}

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
