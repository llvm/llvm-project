//===- Standard pass instrumentations handling ----------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines IR-printing pass instrumentation callbacks as well as
/// StandardInstrumentations class that manages standard pass instrumentations.
///
//===----------------------------------------------------------------------===//

#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

using namespace llvm;

// TODO: remove once all required passes are marked as such.
static cl::opt<bool>
    EnableOptnone("enable-npm-optnone", cl::init(false),
                  cl::desc("Enable skipping optional passes optnone functions "
                           "under new pass manager"));

cl::opt<bool> PreservedCFGCheckerInstrumentation::VerifyPreservedCFG(
    "verify-cfg-preserved", cl::Hidden,
#ifdef NDEBUG
    cl::init(false));
#else
    cl::init(true));
#endif

// FIXME: Change `-debug-pass-manager` from boolean to enum type. Similar to
// `-debug-pass` in legacy PM.
static cl::opt<bool>
    DebugPMVerbose("debug-pass-manager-verbose", cl::Hidden, cl::init(false),
                   cl::desc("Print all pass management debugging information. "
                            "`-debug-pass-manager` must also be specified"));

namespace {

/// Extracting Module out of \p IR unit. Also fills a textual description
/// of \p IR for use in header when printing.
Optional<std::pair<const Module *, std::string>> unwrapModule(Any IR) {
  if (any_isa<const Module *>(IR))
    return std::make_pair(any_cast<const Module *>(IR), std::string());

  if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    if (!llvm::isFunctionInPrintList(F->getName()))
      return None;
    const Module *M = F->getParent();
    return std::make_pair(M, formatv(" (function: {0})", F->getName()).str());
  }

  if (any_isa<const LazyCallGraph::SCC *>(IR)) {
    const LazyCallGraph::SCC *C = any_cast<const LazyCallGraph::SCC *>(IR);
    for (const LazyCallGraph::Node &N : *C) {
      const Function &F = N.getFunction();
      if (!F.isDeclaration() && isFunctionInPrintList(F.getName())) {
        const Module *M = F.getParent();
        return std::make_pair(M, formatv(" (scc: {0})", C->getName()).str());
      }
    }
    return None;
  }

  if (any_isa<const Loop *>(IR)) {
    const Loop *L = any_cast<const Loop *>(IR);
    const Function *F = L->getHeader()->getParent();
    if (!isFunctionInPrintList(F->getName()))
      return None;
    const Module *M = F->getParent();
    std::string LoopName;
    raw_string_ostream ss(LoopName);
    L->getHeader()->printAsOperand(ss, false);
    return std::make_pair(M, formatv(" (loop: {0})", ss.str()).str());
  }

  llvm_unreachable("Unknown IR unit");
}

void printIR(raw_ostream &OS, const Function *F, StringRef Banner,
             StringRef Extra = StringRef(), bool Brief = false) {
  if (Brief) {
    OS << F->getName() << '\n';
    return;
  }

  if (!llvm::isFunctionInPrintList(F->getName()))
    return;
  OS << Banner << Extra << "\n" << static_cast<const Value &>(*F);
}

void printIR(raw_ostream &OS, const Module *M, StringRef Banner,
             StringRef Extra = StringRef(), bool Brief = false) {
  if (Brief) {
    OS << M->getName() << '\n';
    return;
  }

  if (llvm::isFunctionInPrintList("*") || llvm::forcePrintModuleIR()) {
    OS << Banner << Extra << "\n";
    M->print(OS, nullptr, false);
  } else {
    for (const auto &F : M->functions()) {
      printIR(OS, &F, Banner, Extra);
    }
  }
}

void printIR(raw_ostream &OS, const LazyCallGraph::SCC *C, StringRef Banner,
             StringRef Extra = StringRef(), bool Brief = false) {
  if (Brief) {
    OS << *C << '\n';
    return;
  }

  bool BannerPrinted = false;
  for (const LazyCallGraph::Node &N : *C) {
    const Function &F = N.getFunction();
    if (!F.isDeclaration() && llvm::isFunctionInPrintList(F.getName())) {
      if (!BannerPrinted) {
        OS << Banner << Extra << "\n";
        BannerPrinted = true;
      }
      F.print(OS);
    }
  }
}

void printIR(raw_ostream &OS, const Loop *L, StringRef Banner,
             bool Brief = false) {
  if (Brief) {
    OS << *L;
    return;
  }

  const Function *F = L->getHeader()->getParent();
  if (!llvm::isFunctionInPrintList(F->getName()))
    return;
  llvm::printLoop(const_cast<Loop &>(*L), OS, std::string(Banner));
}

/// Generic IR-printing helper that unpacks a pointer to IRUnit wrapped into
/// llvm::Any and does actual print job.
void unwrapAndPrint(raw_ostream &OS, Any IR, StringRef Banner,
                    bool ForceModule = false, bool Brief = false) {
  if (ForceModule) {
    if (auto UnwrappedModule = unwrapModule(IR))
      printIR(OS, UnwrappedModule->first, Banner, UnwrappedModule->second);
    return;
  }

  if (any_isa<const Module *>(IR)) {
    const Module *M = any_cast<const Module *>(IR);
    assert(M && "module should be valid for printing");
    printIR(OS, M, Banner, "", Brief);
    return;
  }

  if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    assert(F && "function should be valid for printing");
    printIR(OS, F, Banner, "", Brief);
    return;
  }

  if (any_isa<const LazyCallGraph::SCC *>(IR)) {
    const LazyCallGraph::SCC *C = any_cast<const LazyCallGraph::SCC *>(IR);
    assert(C && "scc should be valid for printing");
    std::string Extra = std::string(formatv(" (scc: {0})", C->getName()));
    printIR(OS, C, Banner, Extra, Brief);
    return;
  }

  if (any_isa<const Loop *>(IR)) {
    const Loop *L = any_cast<const Loop *>(IR);
    assert(L && "Loop should be valid for printing");
    printIR(OS, L, Banner, Brief);
    return;
  }
  llvm_unreachable("Unknown wrapped IR type");
}

} // namespace

PrintIRInstrumentation::~PrintIRInstrumentation() {
  assert(ModuleDescStack.empty() && "ModuleDescStack is not empty at exit");
}

void PrintIRInstrumentation::pushModuleDesc(StringRef PassID, Any IR) {
  assert(StoreModuleDesc);
  const Module *M = nullptr;
  std::string Extra;
  if (auto UnwrappedModule = unwrapModule(IR))
    std::tie(M, Extra) = UnwrappedModule.getValue();
  ModuleDescStack.emplace_back(M, Extra, PassID);
}

PrintIRInstrumentation::PrintModuleDesc
PrintIRInstrumentation::popModuleDesc(StringRef PassID) {
  assert(!ModuleDescStack.empty() && "empty ModuleDescStack");
  PrintModuleDesc ModuleDesc = ModuleDescStack.pop_back_val();
  assert(std::get<2>(ModuleDesc).equals(PassID) && "malformed ModuleDescStack");
  return ModuleDesc;
}

void PrintIRInstrumentation::printBeforePass(StringRef PassID, Any IR) {
  if (PassID.startswith("PassManager<") || PassID.contains("PassAdaptor<"))
    return;

  // Saving Module for AfterPassInvalidated operations.
  // Note: here we rely on a fact that we do not change modules while
  // traversing the pipeline, so the latest captured module is good
  // for all print operations that has not happen yet.
  if (StoreModuleDesc && llvm::shouldPrintAfterPass(PassID))
    pushModuleDesc(PassID, IR);

  if (!llvm::shouldPrintBeforePass(PassID))
    return;

  SmallString<20> Banner = formatv("*** IR Dump Before {0} ***", PassID);
  unwrapAndPrint(dbgs(), IR, Banner, llvm::forcePrintModuleIR());
  return;
}

void PrintIRInstrumentation::printAfterPass(StringRef PassID, Any IR) {
  if (PassID.startswith("PassManager<") || PassID.contains("PassAdaptor<"))
    return;

  if (!llvm::shouldPrintAfterPass(PassID))
    return;

  if (StoreModuleDesc)
    popModuleDesc(PassID);

  SmallString<20> Banner = formatv("*** IR Dump After {0} ***", PassID);
  unwrapAndPrint(dbgs(), IR, Banner, llvm::forcePrintModuleIR());
}

void PrintIRInstrumentation::printAfterPassInvalidated(StringRef PassID) {
  if (!StoreModuleDesc || !llvm::shouldPrintAfterPass(PassID))
    return;

  if (PassID.startswith("PassManager<") || PassID.contains("PassAdaptor<"))
    return;

  const Module *M;
  std::string Extra;
  StringRef StoredPassID;
  std::tie(M, Extra, StoredPassID) = popModuleDesc(PassID);
  // Additional filtering (e.g. -filter-print-func) can lead to module
  // printing being skipped.
  if (!M)
    return;

  SmallString<20> Banner =
      formatv("*** IR Dump After {0} *** invalidated: ", PassID);
  printIR(dbgs(), M, Banner, Extra);
}

void PrintIRInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  // BeforePass callback is not just for printing, it also saves a Module
  // for later use in AfterPassInvalidated.
  StoreModuleDesc = llvm::forcePrintModuleIR() && llvm::shouldPrintAfterPass();
  if (llvm::shouldPrintBeforePass() || StoreModuleDesc)
    PIC.registerBeforeNonSkippedPassCallback(
        [this](StringRef P, Any IR) { this->printBeforePass(P, IR); });

  if (llvm::shouldPrintAfterPass()) {
    PIC.registerAfterPassCallback(
        [this](StringRef P, Any IR, const PreservedAnalyses &) {
          this->printAfterPass(P, IR);
        });
    PIC.registerAfterPassInvalidatedCallback(
        [this](StringRef P, const PreservedAnalyses &) {
          this->printAfterPassInvalidated(P);
        });
  }
}

void OptNoneInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  PIC.registerBeforePassCallback(
      [this](StringRef P, Any IR) { return this->skip(P, IR); });
}

bool OptNoneInstrumentation::skip(StringRef PassID, Any IR) {
  if (!EnableOptnone)
    return true;
  const Function *F = nullptr;
  if (any_isa<const Function *>(IR)) {
    F = any_cast<const Function *>(IR);
  } else if (any_isa<const Loop *>(IR)) {
    F = any_cast<const Loop *>(IR)->getHeader()->getParent();
  }
  return !(F && F->hasOptNone());
}

void PrintPassInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  if (!DebugLogging)
    return;

  std::vector<StringRef> SpecialPasses = {"PassManager"};
  if (!DebugPMVerbose)
    SpecialPasses.emplace_back("PassAdaptor");

  PIC.registerBeforeSkippedPassCallback(
      [SpecialPasses](StringRef PassID, Any IR) {
        assert(!isSpecialPass(PassID, SpecialPasses) &&
               "Unexpectedly skipping special pass");

        dbgs() << "Skipping pass: " << PassID << " on ";
        unwrapAndPrint(dbgs(), IR, "", false, true);
      });

  PIC.registerBeforeNonSkippedPassCallback(
      [SpecialPasses](StringRef PassID, Any IR) {
        if (isSpecialPass(PassID, SpecialPasses))
          return;

        dbgs() << "Running pass: " << PassID << " on ";
        unwrapAndPrint(dbgs(), IR, "", false, true);
      });

  PIC.registerBeforeAnalysisCallback([](StringRef PassID, Any IR) {
    dbgs() << "Running analysis: " << PassID << " on ";
    unwrapAndPrint(dbgs(), IR, "", false, true);
  });
}

PreservedCFGCheckerInstrumentation::CFG::CFG(const Function *F,
                                             bool TrackBBLifetime) {
  if (TrackBBLifetime)
    BBGuards = DenseMap<intptr_t, BBGuard>(F->size());
  for (const auto &BB : *F) {
    if (BBGuards)
      BBGuards->try_emplace(intptr_t(&BB), &BB);
    for (auto *Succ : successors(&BB)) {
      Graph[&BB][Succ]++;
      if (BBGuards)
        BBGuards->try_emplace(intptr_t(Succ), Succ);
    }
  }
}

static void printBBName(raw_ostream &out, const BasicBlock *BB) {
  if (BB->hasName()) {
    out << BB->getName() << "<" << BB << ">";
    return;
  }

  if (!BB->getParent()) {
    out << "unnamed_removed<" << BB << ">";
    return;
  }

  if (BB == &BB->getParent()->getEntryBlock()) {
    out << "entry"
        << "<" << BB << ">";
    return;
  }

  unsigned FuncOrderBlockNum = 0;
  for (auto &FuncBB : *BB->getParent()) {
    if (&FuncBB == BB)
      break;
    FuncOrderBlockNum++;
  }
  out << "unnamed_" << FuncOrderBlockNum << "<" << BB << ">";
}

void PreservedCFGCheckerInstrumentation::CFG::printDiff(raw_ostream &out,
                                                        const CFG &Before,
                                                        const CFG &After) {
  assert(!After.isPoisoned());

  // Print function name.
  const CFG *FuncGraph = nullptr;
  if (!After.Graph.empty())
    FuncGraph = &After;
  else if (!Before.isPoisoned() && !Before.Graph.empty())
    FuncGraph = &Before;

  if (FuncGraph)
    out << "In function @"
        << FuncGraph->Graph.begin()->first->getParent()->getName() << "\n";

  if (Before.isPoisoned()) {
    out << "Some blocks were deleted\n";
    return;
  }

  // Find and print graph differences.
  if (Before.Graph.size() != After.Graph.size())
    out << "Different number of non-leaf basic blocks: before="
        << Before.Graph.size() << ", after=" << After.Graph.size() << "\n";

  for (auto &BB : Before.Graph) {
    auto BA = After.Graph.find(BB.first);
    if (BA == After.Graph.end()) {
      out << "Non-leaf block ";
      printBBName(out, BB.first);
      out << " is removed (" << BB.second.size() << " successors)\n";
    }
  }

  for (auto &BA : After.Graph) {
    auto BB = Before.Graph.find(BA.first);
    if (BB == Before.Graph.end()) {
      out << "Non-leaf block ";
      printBBName(out, BA.first);
      out << " is added (" << BA.second.size() << " successors)\n";
      continue;
    }

    if (BB->second == BA.second)
      continue;

    out << "Different successors of block ";
    printBBName(out, BA.first);
    out << " (unordered):\n";
    out << "- before (" << BB->second.size() << "): ";
    for (auto &SuccB : BB->second) {
      printBBName(out, SuccB.first);
      if (SuccB.second != 1)
        out << "(" << SuccB.second << "), ";
      else
        out << ", ";
    }
    out << "\n";
    out << "- after (" << BA.second.size() << "): ";
    for (auto &SuccA : BA.second) {
      printBBName(out, SuccA.first);
      if (SuccA.second != 1)
        out << "(" << SuccA.second << "), ";
      else
        out << ", ";
    }
    out << "\n";
  }
}

void PreservedCFGCheckerInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  if (!VerifyPreservedCFG)
    return;

  PIC.registerBeforeNonSkippedPassCallback([this](StringRef P, Any IR) {
    if (any_isa<const Function *>(IR))
      GraphStackBefore.emplace_back(P, CFG(any_cast<const Function *>(IR)));
    else
      GraphStackBefore.emplace_back(P, None);
  });

  PIC.registerAfterPassInvalidatedCallback(
      [this](StringRef P, const PreservedAnalyses &PassPA) {
        auto Before = GraphStackBefore.pop_back_val();
        assert(Before.first == P &&
               "Before and After callbacks must correspond");
        (void)Before;
      });

  PIC.registerAfterPassCallback([this](StringRef P, Any IR,
                                       const PreservedAnalyses &PassPA) {
    auto Before = GraphStackBefore.pop_back_val();
    assert(Before.first == P && "Before and After callbacks must correspond");
    auto &GraphBefore = Before.second;

    if (!PassPA.allAnalysesInSetPreserved<CFGAnalyses>())
      return;

    if (any_isa<const Function *>(IR)) {
      assert(GraphBefore && "Must be built in BeforePassCallback");
      CFG GraphAfter(any_cast<const Function *>(IR), false /* NeedsGuard */);
      if (GraphAfter == *GraphBefore)
        return;

      dbgs() << "Error: " << P
             << " reported it preserved CFG, but changes detected:\n";
      CFG::printDiff(dbgs(), *GraphBefore, GraphAfter);
      report_fatal_error(Twine("Preserved CFG changed by ", P));
    }
  });
}

void StandardInstrumentations::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  PrintIR.registerCallbacks(PIC);
  PrintPass.registerCallbacks(PIC);
  TimePasses.registerCallbacks(PIC);
  OptNone.registerCallbacks(PIC);
  PreservedCFGChecker.registerCallbacks(PIC);
}
