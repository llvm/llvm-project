//===-- GCMetadata.cpp - Garbage collector metadata -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GCFunctionInfo class and GCModuleInfo pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <string>

using namespace llvm;

namespace {

class Printer : public FunctionPass {
  static char ID;

  raw_ostream &OS;

public:
  explicit Printer(raw_ostream &OS) : FunctionPass(ID), OS(OS) {}

  StringRef getPassName() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;
  bool doFinalization(Module &M) override;
};

} // end anonymous namespace

INITIALIZE_PASS(GCModuleInfo, "collector-metadata",
                "Create Garbage Collector Module Metadata", false, false)

// -----------------------------------------------------------------------------

GCFunctionInfo::GCFunctionInfo(const Function &F, GCStrategy &S)
    : F(F), S(S), FrameSize(~0LL) {}

GCFunctionInfo::~GCFunctionInfo() = default;

// -----------------------------------------------------------------------------

char GCModuleInfo::ID = 0;

GCModuleInfo::GCModuleInfo() : ImmutablePass(ID) {
  initializeGCModuleInfoPass(*PassRegistry::getPassRegistry());
}

GCFunctionInfo &GCModuleInfo::getFunctionInfo(const Function &F) {
  assert(!F.isDeclaration() && "Can only get GCFunctionInfo for a definition!");
  assert(F.hasGC());

  finfo_map_type::iterator I = FInfoMap.find(&F);
  if (I != FInfoMap.end())
    return *I->second;

  GCStrategy *S = getGCStrategy(F.getGC());
  Functions.push_back(std::make_unique<GCFunctionInfo>(F, *S));
  GCFunctionInfo *GFI = Functions.back().get();
  FInfoMap[&F] = GFI;
  return *GFI;
}

void GCModuleInfo::clear() {
  Functions.clear();
  FInfoMap.clear();
  GCStrategyList.clear();
}

// -----------------------------------------------------------------------------

char Printer::ID = 0;

FunctionPass *llvm::createGCInfoPrinter(raw_ostream &OS) {
  return new Printer(OS);
}

StringRef Printer::getPassName() const {
  return "Print Garbage Collector Information";
}

void Printer::getAnalysisUsage(AnalysisUsage &AU) const {
  FunctionPass::getAnalysisUsage(AU);
  AU.setPreservesAll();
  AU.addRequired<GCModuleInfo>();
}

bool Printer::runOnFunction(Function &F) {
  if (F.hasGC())
    return false;

  GCFunctionInfo *FD = &getAnalysis<GCModuleInfo>().getFunctionInfo(F);

  OS << "GC roots for " << FD->getFunction().getName() << ":\n";
  for (GCFunctionInfo::roots_iterator RI = FD->roots_begin(),
                                      RE = FD->roots_end();
       RI != RE; ++RI)
    OS << "\t" << RI->Num << "\t" << RI->StackOffset << "[sp]\n";

  OS << "GC safe points for " << FD->getFunction().getName() << ":\n";
  for (GCFunctionInfo::iterator PI = FD->begin(), PE = FD->end(); PI != PE;
       ++PI) {

    OS << "\t" << PI->Label->getName() << ": " << "post-call"
       << ", live = {";

    ListSeparator LS(",");
    for (const GCRoot &R : make_range(FD->live_begin(PI), FD->live_end(PI)))
      OS << LS << " " << R.Num;

    OS << " }\n";
  }

  return false;
}

bool Printer::doFinalization(Module &M) {
  GCModuleInfo *GMI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(GMI && "Printer didn't require GCModuleInfo?!");
  GMI->clear();
  return false;
}

GCStrategy *GCModuleInfo::getGCStrategy(const StringRef Name) {
  // TODO: Arguably, just doing a linear search would be faster for small N
  auto NMI = GCStrategyMap.find(Name);
  if (NMI != GCStrategyMap.end())
    return NMI->getValue();

  std::unique_ptr<GCStrategy> S = llvm::getGCStrategy(Name);
  S->Name = std::string(Name);
  GCStrategyMap[Name] = S.get();
  GCStrategyList.push_back(std::move(S));
  return GCStrategyList.back().get();
}
