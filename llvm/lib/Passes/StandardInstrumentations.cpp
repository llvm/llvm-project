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
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>
#include <vector>

using namespace llvm;

// TODO: remove once all required passes are marked as such.
static cl::opt<bool>
    EnableOptnone("enable-npm-optnone", cl::init(true),
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

// An option that prints out the IR after passes, similar to
// -print-after-all except that it only prints the IR after passes that
// change the IR.  Those passes that do not make changes to the IR are
// reported as not making any changes.  In addition, the initial IR is
// also reported.  Other hidden options affect the output from this
// option.  -filter-passes will limit the output to the named passes
// that actually change the IR and other passes are reported as filtered out.
// The specified passes will either be reported as making no changes (with
// no IR reported) or the changed IR will be reported.  Also, the
// -filter-print-funcs and -print-module-scope options will do similar
// filtering based on function name, reporting changed IRs as functions(or
// modules if -print-module-scope is specified) for a particular function
// or indicating that the IR has been filtered out.  The extra options
// can be combined, allowing only changed IRs for certain passes on certain
// functions to be reported in different formats, with the rest being
// reported as filtered out.
static cl::opt<bool> PrintChanged("print-changed",
                                  cl::desc("Print changed IRs"),
                                  cl::init(false), cl::Hidden);
// An option that supports the -print-changed option.  See
// the description for -print-changed for an explanation of the use
// of this option.  Note that this option has no effect without -print-changed.
static cl::list<std::string>
    PrintPassesList("filter-passes", cl::value_desc("pass names"),
                    cl::desc("Only consider IR changes for passes whose names "
                             "match for the print-changed option"),
                    cl::CommaSeparated, cl::Hidden);

namespace {

/// Extracting Module out of \p IR unit. Also fills a textual description
/// of \p IR for use in header when printing.
Optional<std::pair<const Module *, std::string>>
unwrapModule(Any IR, bool Force = false) {
  if (any_isa<const Module *>(IR))
    return std::make_pair(any_cast<const Module *>(IR), std::string());

  if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    if (!Force && !llvm::isFunctionInPrintList(F->getName()))
      return None;

    const Module *M = F->getParent();
    return std::make_pair(M, formatv(" (function: {0})", F->getName()).str());
  }

  if (any_isa<const LazyCallGraph::SCC *>(IR)) {
    const LazyCallGraph::SCC *C = any_cast<const LazyCallGraph::SCC *>(IR);
    for (const LazyCallGraph::Node &N : *C) {
      const Function &F = N.getFunction();
      if (Force || (!F.isDeclaration() && isFunctionInPrintList(F.getName()))) {
        const Module *M = F.getParent();
        return std::make_pair(M, formatv(" (scc: {0})", C->getName()).str());
      }
    }
    assert(!Force && "Expected to have made a pair when forced.");
    return None;
  }

  if (any_isa<const Loop *>(IR)) {
    const Loop *L = any_cast<const Loop *>(IR);
    const Function *F = L->getHeader()->getParent();
    if (!Force && !isFunctionInPrintList(F->getName()))
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
             StringRef Extra = StringRef(), bool Brief = false,
             bool ShouldPreserveUseListOrder = false) {
  if (Brief) {
    OS << M->getName() << '\n';
    return;
  }

  if (llvm::isFunctionInPrintList("*") || llvm::forcePrintModuleIR()) {
    OS << Banner << Extra << "\n";
    M->print(OS, nullptr, ShouldPreserveUseListOrder);
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
                    bool ForceModule = false, bool Brief = false,
                    bool ShouldPreserveUseListOrder = false) {
  if (ForceModule) {
    if (auto UnwrappedModule = unwrapModule(IR))
      printIR(OS, UnwrappedModule->first, Banner, UnwrappedModule->second,
              Brief, ShouldPreserveUseListOrder);
    return;
  }

  if (any_isa<const Module *>(IR)) {
    const Module *M = any_cast<const Module *>(IR);
    assert(M && "module should be valid for printing");
    printIR(OS, M, Banner, "", Brief, ShouldPreserveUseListOrder);
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

// Return true when this is a pass for which changes should be ignored
bool isIgnored(StringRef PassID) {
  return isSpecialPass(PassID,
                       {"PassManager", "PassAdaptor", "AnalysisManagerProxy"});
}

// Return true when this is a defined function for which printing
// of changes is desired.
bool isInterestingFunction(const Function &F) {
  return llvm::isFunctionInPrintList(F.getName());
}

// Return true when this is a pass for which printing of changes is desired.
bool isInterestingPass(StringRef PassID) {
  if (isIgnored(PassID))
    return false;

  static std::unordered_set<std::string> PrintPassNames(PrintPassesList.begin(),
                                                        PrintPassesList.end());
  return PrintPassNames.empty() || PrintPassNames.count(PassID.str());
}

// Return true when this is a pass on IR for which printing
// of changes is desired.
bool isInteresting(Any IR, StringRef PassID) {
  if (!isInterestingPass(PassID))
    return false;
  if (any_isa<const Function *>(IR))
    return isInterestingFunction(*any_cast<const Function *>(IR));
  return true;
}

} // namespace

template <typename IRUnitT>
void ChangePrinter<IRUnitT>::saveIRBeforePass(Any IR, StringRef PassID) {
  // Always need to place something on the stack because invalidated passes
  // are not given the IR so it cannot be determined whether the pass was for
  // something that was filtered out.
  BeforeStack.emplace_back();

  if (!isInteresting(IR, PassID))
    return;
  // Is this the initial IR?
  if (InitialIR) {
    InitialIR = false;
    handleInitialIR(IR);
  }

  // Save the IR representation on the stack.
  IRUnitT &Data = BeforeStack.back();
  generateIRRepresentation(IR, PassID, Data);
}

template <typename IRUnitT>
void ChangePrinter<IRUnitT>::handleIRAfterPass(Any IR, StringRef PassID) {
  assert(!BeforeStack.empty() && "Unexpected empty stack encountered.");
  std::string Name;

  // unwrapModule has inconsistent handling of names for function IRs.
  if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    Name = formatv(" (function: {0})", F->getName()).str();
  } else {
    if (auto UM = unwrapModule(IR))
      Name = UM->second;
  }
  if (Name.empty())
    Name = " (module)";

  if (isIgnored(PassID))
    handleIgnored(PassID, Name);
  else if (!isInteresting(IR, PassID))
    handleFiltered(PassID, Name);
  else {
    // Get the before rep from the stack
    IRUnitT &Before = BeforeStack.back();
    // Create the after rep
    IRUnitT After;
    generateIRRepresentation(IR, PassID, After);

    // Was there a change in IR?
    if (same(Before, After))
      omitAfter(PassID, Name);
    else
      handleAfter(PassID, Name, Before, After, IR);
  }
  BeforeStack.pop_back();
}

template <typename IRUnitT>
void ChangePrinter<IRUnitT>::handleInvalidatedPass(StringRef PassID) {
  assert(!BeforeStack.empty() && "Unexpected empty stack encountered.");

  // Always flag it as invalidated as we cannot determine when
  // a pass for a filtered function is invalidated since we do not
  // get the IR in the call.  Also, the output is just alternate
  // forms of the banner anyway.
  handleInvalidated(PassID);
  BeforeStack.pop_back();
}

template <typename IRUnitT> ChangePrinter<IRUnitT>::~ChangePrinter<IRUnitT>() {
  assert(BeforeStack.empty() && "Problem with Change Printer stack.");
}

IRChangePrinter::IRChangePrinter() : Out(dbgs()) {}

IRChangePrinter::~IRChangePrinter() {}

void IRChangePrinter::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (!PrintChanged)
    return;

  PIC.registerBeforePassCallback([this](StringRef P, Any IR) {
    saveIRBeforePass(IR, P);
    return true;
  });

  PIC.registerAfterPassCallback(
      [this](StringRef P, Any IR, const PreservedAnalyses &) {
        handleIRAfterPass(IR, P);
      });
  PIC.registerAfterPassInvalidatedCallback(
      [this](StringRef P, const PreservedAnalyses &) {
        handleInvalidatedPass(P);
      });
}

void IRChangePrinter::handleInitialIR(Any IR) {
  // Always print the module.
  // Unwrap and print directly to avoid filtering problems in general routines.
  auto UnwrappedModule = unwrapModule(IR, /*Force=*/true);
  assert(UnwrappedModule && "Expected module to be unwrapped when forced.");
  Out << "*** IR Dump At Start: ***" << UnwrappedModule->second << "\n";
  UnwrappedModule->first->print(Out, nullptr,
                                /*ShouldPreserveUseListOrder=*/true);
}

void IRChangePrinter::generateIRRepresentation(Any IR, StringRef PassID,
                                               std::string &Output) {
  raw_string_ostream OS(Output);
  // use the after banner for all cases so it will match
  SmallString<20> Banner = formatv("*** IR Dump After {0} ***", PassID);
  unwrapAndPrint(OS, IR, Banner, llvm::forcePrintModuleIR(),
                 /*Brief=*/false, /*ShouldPreserveUseListOrder=*/true);
  OS.str();
}

void IRChangePrinter::omitAfter(StringRef PassID, std::string &Name) {
  Out << formatv("*** IR Dump After {0}{1} omitted because no change ***\n",
                 PassID, Name);
}

void IRChangePrinter::handleAfter(StringRef PassID, std::string &Name,
                                  const std::string &Before,
                                  const std::string &After, Any) {
  assert(After.find("*** IR Dump") == 0 && "Unexpected banner format.");
  StringRef AfterRef = After;
  StringRef Banner =
      AfterRef.take_until([](char C) -> bool { return C == '\n'; });
  Out << Banner;

  // LazyCallGraph::SCC already has "(scc:..." in banner so only add
  // in the name if it isn't already there.
  if (Name.substr(0, 6) != " (scc:" && !llvm::forcePrintModuleIR())
    Out << Name;

  Out << After.substr(Banner.size());
}

void IRChangePrinter::handleInvalidated(StringRef PassID) {
  Out << formatv("*** IR Pass {0} invalidated ***\n", PassID);
}

void IRChangePrinter::handleFiltered(StringRef PassID, std::string &Name) {
  SmallString<20> Banner =
      formatv("*** IR Dump After {0}{1} filtered out ***\n", PassID, Name);
  Out << Banner;
}

void IRChangePrinter::handleIgnored(StringRef PassID, std::string &Name) {
  Out << formatv("*** IR Pass {0}{1} ignored ***\n", PassID, Name);
}

bool IRChangePrinter::same(const std::string &Before,
                           const std::string &After) {
  return Before == After;
}

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

void VerifyInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  PIC.registerAfterPassCallback(
      [this](StringRef P, Any IR, const PreservedAnalyses &PassPA) {
        if (isIgnored(P) || P == "VerifierPass")
          return;
        if (any_isa<const Function *>(IR) || any_isa<const Loop *>(IR)) {
          const Function *F;
          if (any_isa<const Loop *>(IR))
            F = any_cast<const Loop *>(IR)->getHeader()->getParent();
          else
            F = any_cast<const Function *>(IR);
          if (DebugLogging)
            dbgs() << "Verifying function " << F->getName() << "\n";

          if (verifyFunction(*F))
            report_fatal_error("Broken function found, compilation aborted!");
        } else if (any_isa<const Module *>(IR) ||
                   any_isa<const LazyCallGraph::SCC *>(IR)) {
          const Module *M;
          if (any_isa<const LazyCallGraph::SCC *>(IR))
            M = any_cast<const LazyCallGraph::SCC *>(IR)
                    ->begin()
                    ->getFunction()
                    .getParent();
          else
            M = any_cast<const Module *>(IR);
          if (DebugLogging)
            dbgs() << "Verifying module " << M->getName() << "\n";

          if (verifyModule(*M))
            report_fatal_error("Broken module found, compilation aborted!");
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
  PrintChangedIR.registerCallbacks(PIC);
  if (VerifyEach)
    Verify.registerCallbacks(PIC);
}
