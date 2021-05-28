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
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>
#include <vector>

using namespace llvm;

cl::opt<bool> PreservedCFGCheckerInstrumentation::VerifyPreservedCFG(
    "verify-cfg-preserved", cl::Hidden,
#ifdef NDEBUG
    cl::init(false));
#else
    cl::init(true));
#endif

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
// reported as filtered out.  The -print-before-changed option will print
// the IR as it was before each pass that changed it.  The optional
// value of quiet will only report when the IR changes, suppressing
// all other messages, including the initial IR.  The values "diff" and
// "diff-quiet" will present the changes in a form similar to a patch, in
// either verbose or quiet mode, respectively.  The lines that are removed
// and added are prefixed with '-' and '+', respectively.  The
// -filter-print-funcs and -filter-passes can be used to filter the output.
// This reporter relies on the linux diff utility to do comparisons and
// insert the prefixes.  For systems that do not have the necessary
// facilities, the error message will be shown in place of the expected output.
//
enum class ChangePrinter {
  NoChangePrinter,
  PrintChangedVerbose,
  PrintChangedQuiet,
  PrintChangedDiffVerbose,
  PrintChangedDiffQuiet,
  PrintChangedColourDiffVerbose,
  PrintChangedColourDiffQuiet
};
static cl::opt<ChangePrinter> PrintChanged(
    "print-changed", cl::desc("Print changed IRs"), cl::Hidden,
    cl::ValueOptional, cl::init(ChangePrinter::NoChangePrinter),
    cl::values(
        clEnumValN(ChangePrinter::PrintChangedQuiet, "quiet",
                   "Run in quiet mode"),
        clEnumValN(ChangePrinter::PrintChangedDiffVerbose, "diff",
                   "Display patch-like changes"),
        clEnumValN(ChangePrinter::PrintChangedDiffQuiet, "diff-quiet",
                   "Display patch-like changes in quiet mode"),
        clEnumValN(ChangePrinter::PrintChangedColourDiffVerbose, "cdiff",
                   "Display patch-like changes with color"),
        clEnumValN(ChangePrinter::PrintChangedColourDiffQuiet, "cdiff-quiet",
                   "Display patch-like changes in quiet mode with color"),
        // Sentinel value for unspecified option.
        clEnumValN(ChangePrinter::PrintChangedVerbose, "", "")));

// An option that supports the -print-changed option.  See
// the description for -print-changed for an explanation of the use
// of this option.  Note that this option has no effect without -print-changed.
static cl::list<std::string>
    PrintPassesList("filter-passes", cl::value_desc("pass names"),
                    cl::desc("Only consider IR changes for passes whose names "
                             "match for the print-changed option"),
                    cl::CommaSeparated, cl::Hidden);
// An option that supports the -print-changed option.  See
// the description for -print-changed for an explanation of the use
// of this option.  Note that this option has no effect without -print-changed.
static cl::opt<bool>
    PrintChangedBefore("print-before-changed",
                       cl::desc("Print before passes that change them"),
                       cl::init(false), cl::Hidden);

// An option for specifying the diff used by print-changed=[diff | diff-quiet]
static cl::opt<std::string>
    DiffBinary("print-changed-diff-path", cl::Hidden, cl::init("diff"),
               cl::desc("system diff used by change reporters"));

namespace {

// Perform a system based diff between \p Before and \p After, using
// \p OldLineFormat, \p NewLineFormat, and \p UnchangedLineFormat
// to control the formatting of the output.  Return an error message
// for any failures instead of the diff.
std::string doSystemDiff(StringRef Before, StringRef After,
                         StringRef OldLineFormat, StringRef NewLineFormat,
                         StringRef UnchangedLineFormat) {
  StringRef SR[2]{Before, After};
  // Store the 2 bodies into temporary files and call diff on them
  // to get the body of the node.
  const unsigned NumFiles = 3;
  static std::string FileName[NumFiles];
  static int FD[NumFiles]{-1, -1, -1};
  for (unsigned I = 0; I < NumFiles; ++I) {
    if (FD[I] == -1) {
      SmallVector<char, 200> SV;
      std::error_code EC =
          sys::fs::createTemporaryFile("tmpdiff", "txt", FD[I], SV);
      if (EC)
        return "Unable to create temporary file.";
      FileName[I] = Twine(SV).str();
    }
    // The third file is used as the result of the diff.
    if (I == NumFiles - 1)
      break;

    std::error_code EC = sys::fs::openFileForWrite(FileName[I], FD[I]);
    if (EC)
      return "Unable to open temporary file for writing.";

    raw_fd_ostream OutStream(FD[I], /*shouldClose=*/true);
    if (FD[I] == -1)
      return "Error opening file for writing.";
    OutStream << SR[I];
  }

  static ErrorOr<std::string> DiffExe = sys::findProgramByName(DiffBinary);
  if (!DiffExe)
    return "Unable to find diff executable.";

  SmallString<128> OLF = formatv("--old-line-format={0}", OldLineFormat);
  SmallString<128> NLF = formatv("--new-line-format={0}", NewLineFormat);
  SmallString<128> ULF =
      formatv("--unchanged-line-format={0}", UnchangedLineFormat);

  StringRef Args[] = {"-w", "-d", OLF, NLF, ULF, FileName[0], FileName[1]};
  Optional<StringRef> Redirects[] = {None, StringRef(FileName[2]), None};
  int Result = sys::ExecuteAndWait(*DiffExe, Args, None, Redirects);
  if (Result < 0)
    return "Error executing system diff.";
  std::string Diff;
  auto B = MemoryBuffer::getFile(FileName[2]);
  if (B && *B)
    Diff = (*B)->getBuffer().str();
  else
    return "Unable to read result.";

  // Clean up.
  for (unsigned I = 0; I < NumFiles; ++I) {
    std::error_code EC = sys::fs::remove(FileName[I]);
    if (EC)
      return "Unable to remove temporary file.";
  }
  return Diff;
}

/// Extract Module out of \p IR unit. May return nullptr if \p IR does not match
/// certain global filters. Will never return nullptr if \p Force is true.
const Module *unwrapModule(Any IR, bool Force = false) {
  if (any_isa<const Module *>(IR))
    return any_cast<const Module *>(IR);

  if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    if (!Force && !isFunctionInPrintList(F->getName()))
      return nullptr;

    return F->getParent();
  }

  if (any_isa<const LazyCallGraph::SCC *>(IR)) {
    const LazyCallGraph::SCC *C = any_cast<const LazyCallGraph::SCC *>(IR);
    for (const LazyCallGraph::Node &N : *C) {
      const Function &F = N.getFunction();
      if (Force || (!F.isDeclaration() && isFunctionInPrintList(F.getName()))) {
        return F.getParent();
      }
    }
    assert(!Force && "Expected a module");
    return nullptr;
  }

  if (any_isa<const Loop *>(IR)) {
    const Loop *L = any_cast<const Loop *>(IR);
    const Function *F = L->getHeader()->getParent();
    if (!Force && !isFunctionInPrintList(F->getName()))
      return nullptr;
    return F->getParent();
  }

  llvm_unreachable("Unknown IR unit");
}

void printIR(raw_ostream &OS, const Function *F) {
  if (!isFunctionInPrintList(F->getName()))
    return;
  OS << *F;
}

void printIR(raw_ostream &OS, const Module *M,
             bool ShouldPreserveUseListOrder = false) {
  if (isFunctionInPrintList("*") || forcePrintModuleIR()) {
    M->print(OS, nullptr, ShouldPreserveUseListOrder);
  } else {
    for (const auto &F : M->functions()) {
      printIR(OS, &F);
    }
  }
}

void printIR(raw_ostream &OS, const LazyCallGraph::SCC *C) {
  for (const LazyCallGraph::Node &N : *C) {
    const Function &F = N.getFunction();
    if (!F.isDeclaration() && isFunctionInPrintList(F.getName())) {
      F.print(OS);
    }
  }
}

void printIR(raw_ostream &OS, const Loop *L) {
  const Function *F = L->getHeader()->getParent();
  if (!isFunctionInPrintList(F->getName()))
    return;
  printLoop(const_cast<Loop &>(*L), OS);
}

std::string getIRName(Any IR) {
  if (any_isa<const Module *>(IR))
    return "[module]";

  if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    return F->getName().str();
  }

  if (any_isa<const LazyCallGraph::SCC *>(IR)) {
    const LazyCallGraph::SCC *C = any_cast<const LazyCallGraph::SCC *>(IR);
    return C->getName();
  }

  if (any_isa<const Loop *>(IR)) {
    const Loop *L = any_cast<const Loop *>(IR);
    std::string S;
    raw_string_ostream OS(S);
    L->print(OS, /*Verbose*/ false, /*PrintNested*/ false);
    return OS.str();
  }

  llvm_unreachable("Unknown wrapped IR type");
}

bool moduleContainsFilterPrintFunc(const Module &M) {
  return any_of(M.functions(),
                [](const Function &F) {
                  return isFunctionInPrintList(F.getName());
                }) ||
         isFunctionInPrintList("*");
}

bool sccContainsFilterPrintFunc(const LazyCallGraph::SCC &C) {
  return any_of(C,
                [](const LazyCallGraph::Node &N) {
                  return isFunctionInPrintList(N.getName());
                }) ||
         isFunctionInPrintList("*");
}

bool shouldPrintIR(Any IR) {
  if (any_isa<const Module *>(IR)) {
    const Module *M = any_cast<const Module *>(IR);
    return moduleContainsFilterPrintFunc(*M);
  }

  if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    return isFunctionInPrintList(F->getName());
  }

  if (any_isa<const LazyCallGraph::SCC *>(IR)) {
    const LazyCallGraph::SCC *C = any_cast<const LazyCallGraph::SCC *>(IR);
    return sccContainsFilterPrintFunc(*C);
  }

  if (any_isa<const Loop *>(IR)) {
    const Loop *L = any_cast<const Loop *>(IR);
    return isFunctionInPrintList(L->getHeader()->getParent()->getName());
  }
  llvm_unreachable("Unknown wrapped IR type");
}

/// Generic IR-printing helper that unpacks a pointer to IRUnit wrapped into
/// llvm::Any and does actual print job.
void unwrapAndPrint(raw_ostream &OS, Any IR,
                    bool ShouldPreserveUseListOrder = false) {
  if (!shouldPrintIR(IR))
    return;

  if (forcePrintModuleIR()) {
    auto *M = unwrapModule(IR);
    assert(M && "should have unwrapped module");
    printIR(OS, M, ShouldPreserveUseListOrder);
    return;
  }

  if (any_isa<const Module *>(IR)) {
    const Module *M = any_cast<const Module *>(IR);
    printIR(OS, M, ShouldPreserveUseListOrder);
    return;
  }

  if (any_isa<const Function *>(IR)) {
    const Function *F = any_cast<const Function *>(IR);
    printIR(OS, F);
    return;
  }

  if (any_isa<const LazyCallGraph::SCC *>(IR)) {
    const LazyCallGraph::SCC *C = any_cast<const LazyCallGraph::SCC *>(IR);
    printIR(OS, C);
    return;
  }

  if (any_isa<const Loop *>(IR)) {
    const Loop *L = any_cast<const Loop *>(IR);
    printIR(OS, L);
    return;
  }
  llvm_unreachable("Unknown wrapped IR type");
}

// Return true when this is a pass for which changes should be ignored
bool isIgnored(StringRef PassID) {
  return isSpecialPass(PassID,
                       {"PassManager", "PassAdaptor", "AnalysisManagerProxy",
                        "DevirtSCCRepeatedPass", "ModuleInlinerWrapperPass"});
}

} // namespace

template <typename IRUnitT>
ChangeReporter<IRUnitT>::~ChangeReporter<IRUnitT>() {
  assert(BeforeStack.empty() && "Problem with Change Printer stack.");
}

template <typename IRUnitT>
bool ChangeReporter<IRUnitT>::isInterestingFunction(const Function &F) {
  return isFunctionInPrintList(F.getName());
}

template <typename IRUnitT>
bool ChangeReporter<IRUnitT>::isInterestingPass(StringRef PassID) {
  if (isIgnored(PassID))
    return false;

  static std::unordered_set<std::string> PrintPassNames(PrintPassesList.begin(),
                                                        PrintPassesList.end());
  return PrintPassNames.empty() || PrintPassNames.count(PassID.str());
}

// Return true when this is a pass on IR for which printing
// of changes is desired.
template <typename IRUnitT>
bool ChangeReporter<IRUnitT>::isInteresting(Any IR, StringRef PassID) {
  if (!isInterestingPass(PassID))
    return false;
  if (any_isa<const Function *>(IR))
    return isInterestingFunction(*any_cast<const Function *>(IR));
  return true;
}

template <typename IRUnitT>
void ChangeReporter<IRUnitT>::saveIRBeforePass(Any IR, StringRef PassID) {
  // Always need to place something on the stack because invalidated passes
  // are not given the IR so it cannot be determined whether the pass was for
  // something that was filtered out.
  BeforeStack.emplace_back();

  if (!isInteresting(IR, PassID))
    return;
  // Is this the initial IR?
  if (InitialIR) {
    InitialIR = false;
    if (VerboseMode)
      handleInitialIR(IR);
  }

  // Save the IR representation on the stack.
  IRUnitT &Data = BeforeStack.back();
  generateIRRepresentation(IR, PassID, Data);
}

template <typename IRUnitT>
void ChangeReporter<IRUnitT>::handleIRAfterPass(Any IR, StringRef PassID) {
  assert(!BeforeStack.empty() && "Unexpected empty stack encountered.");

  std::string Name = getIRName(IR);

  if (isIgnored(PassID)) {
    if (VerboseMode)
      handleIgnored(PassID, Name);
  } else if (!isInteresting(IR, PassID)) {
    if (VerboseMode)
      handleFiltered(PassID, Name);
  } else {
    // Get the before rep from the stack
    IRUnitT &Before = BeforeStack.back();
    // Create the after rep
    IRUnitT After;
    generateIRRepresentation(IR, PassID, After);

    // Was there a change in IR?
    if (same(Before, After)) {
      if (VerboseMode)
        omitAfter(PassID, Name);
    } else
      handleAfter(PassID, Name, Before, After, IR);
  }
  BeforeStack.pop_back();
}

template <typename IRUnitT>
void ChangeReporter<IRUnitT>::handleInvalidatedPass(StringRef PassID) {
  assert(!BeforeStack.empty() && "Unexpected empty stack encountered.");

  // Always flag it as invalidated as we cannot determine when
  // a pass for a filtered function is invalidated since we do not
  // get the IR in the call.  Also, the output is just alternate
  // forms of the banner anyway.
  if (VerboseMode)
    handleInvalidated(PassID);
  BeforeStack.pop_back();
}

template <typename IRUnitT>
void ChangeReporter<IRUnitT>::registerRequiredCallbacks(
    PassInstrumentationCallbacks &PIC) {
  PIC.registerBeforeNonSkippedPassCallback(
      [this](StringRef P, Any IR) { saveIRBeforePass(IR, P); });

  PIC.registerAfterPassCallback(
      [this](StringRef P, Any IR, const PreservedAnalyses &) {
        handleIRAfterPass(IR, P);
      });
  PIC.registerAfterPassInvalidatedCallback(
      [this](StringRef P, const PreservedAnalyses &) {
        handleInvalidatedPass(P);
      });
}

ChangedBlockData::ChangedBlockData(const BasicBlock &B)
    : Label(B.getName().str()) {
  raw_string_ostream SS(Body);
  B.print(SS, nullptr, true, true);
}

template <typename IRUnitT>
TextChangeReporter<IRUnitT>::TextChangeReporter(bool Verbose)
    : ChangeReporter<IRUnitT>(Verbose), Out(dbgs()) {}

template <typename IRUnitT>
void TextChangeReporter<IRUnitT>::handleInitialIR(Any IR) {
  // Always print the module.
  // Unwrap and print directly to avoid filtering problems in general routines.
  auto *M = unwrapModule(IR, /*Force=*/true);
  assert(M && "Expected module to be unwrapped when forced.");
  Out << "*** IR Dump At Start ***\n";
  M->print(Out, nullptr,
           /*ShouldPreserveUseListOrder=*/true);
}

template <typename IRUnitT>
void TextChangeReporter<IRUnitT>::omitAfter(StringRef PassID,
                                            std::string &Name) {
  Out << formatv("*** IR Dump After {0} on {1} omitted because no change ***\n",
                 PassID, Name);
}

template <typename IRUnitT>
void TextChangeReporter<IRUnitT>::handleInvalidated(StringRef PassID) {
  Out << formatv("*** IR Pass {0} invalidated ***\n", PassID);
}

template <typename IRUnitT>
void TextChangeReporter<IRUnitT>::handleFiltered(StringRef PassID,
                                                 std::string &Name) {
  SmallString<20> Banner =
      formatv("*** IR Dump After {0} on {1} filtered out ***\n", PassID, Name);
  Out << Banner;
}

template <typename IRUnitT>
void TextChangeReporter<IRUnitT>::handleIgnored(StringRef PassID,
                                                std::string &Name) {
  Out << formatv("*** IR Pass {0} on {1} ignored ***\n", PassID, Name);
}

IRChangedPrinter::~IRChangedPrinter() {}

void IRChangedPrinter::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (PrintChanged == ChangePrinter::PrintChangedVerbose ||
      PrintChanged == ChangePrinter::PrintChangedQuiet)
    TextChangeReporter<std::string>::registerRequiredCallbacks(PIC);
}

void IRChangedPrinter::generateIRRepresentation(Any IR, StringRef PassID,
                                                std::string &Output) {
  raw_string_ostream OS(Output);
  unwrapAndPrint(OS, IR,
                 /*ShouldPreserveUseListOrder=*/true);
  OS.str();
}

void IRChangedPrinter::handleAfter(StringRef PassID, std::string &Name,
                                   const std::string &Before,
                                   const std::string &After, Any) {
  // Report the IR before the changes when requested.
  if (PrintChangedBefore)
    Out << "*** IR Dump Before " << PassID << " on " << Name << " ***\n"
        << Before;

  // We might not get anything to print if we only want to print a specific
  // function but it gets deleted.
  if (After.empty()) {
    Out << "*** IR Deleted After " << PassID << " on " << Name << " ***\n";
    return;
  }

  Out << "*** IR Dump After " << PassID << " on " << Name << " ***\n" << After;
}

bool IRChangedPrinter::same(const std::string &S1, const std::string &S2) {
  return S1 == S2;
}

template <typename IRData>
void OrderedChangedData<IRData>::report(
    const OrderedChangedData &Before, const OrderedChangedData &After,
    function_ref<void(const IRData *, const IRData *)> HandlePair) {
  const auto &BFD = Before.getData();
  const auto &AFD = After.getData();
  std::vector<std::string>::const_iterator BI = Before.getOrder().begin();
  std::vector<std::string>::const_iterator BE = Before.getOrder().end();
  std::vector<std::string>::const_iterator AI = After.getOrder().begin();
  std::vector<std::string>::const_iterator AE = After.getOrder().end();

  auto handlePotentiallyRemovedIRData = [&](std::string S) {
    // The order in LLVM may have changed so check if still exists.
    if (!AFD.count(S)) {
      // This has been removed.
      HandlePair(&BFD.find(*BI)->getValue(), nullptr);
    }
  };
  auto handleNewIRData = [&](std::vector<const IRData *> &Q) {
    // Print out any queued up new sections
    for (const IRData *NBI : Q)
      HandlePair(nullptr, NBI);
    Q.clear();
  };

  // Print out the IRData in the after order, with before ones interspersed
  // appropriately (ie, somewhere near where they were in the before list).
  // Start at the beginning of both lists.  Loop through the
  // after list.  If an element is common, then advance in the before list
  // reporting the removed ones until the common one is reached.  Report any
  // queued up new ones and then report the common one.  If an element is not
  // common, then enqueue it for reporting.  When the after list is exhausted,
  // loop through the before list, reporting any removed ones.  Finally,
  // report the rest of the enqueued new ones.
  std::vector<const IRData *> NewIRDataQueue;
  while (AI != AE) {
    if (!BFD.count(*AI)) {
      // This section is new so place it in the queue.  This will cause it
      // to be reported after deleted sections.
      NewIRDataQueue.emplace_back(&AFD.find(*AI)->getValue());
      ++AI;
      continue;
    }
    // This section is in both; advance and print out any before-only
    // until we get to it.
    while (*BI != *AI) {
      handlePotentiallyRemovedIRData(*BI);
      ++BI;
    }
    // Report any new sections that were queued up and waiting.
    handleNewIRData(NewIRDataQueue);

    const IRData &AData = AFD.find(*AI)->getValue();
    const IRData &BData = BFD.find(*AI)->getValue();
    HandlePair(&BData, &AData);
    ++BI;
    ++AI;
  }

  // Check any remaining before sections to see if they have been removed
  while (BI != BE) {
    handlePotentiallyRemovedIRData(*BI);
    ++BI;
  }

  handleNewIRData(NewIRDataQueue);
}

void ChangedIRComparer::compare(Any IR, StringRef Prefix, StringRef PassID,
                                StringRef Name) {
  if (!getModuleForComparison(IR)) {
    // Not a module so just handle the single function.
    assert(Before.getData().size() == 1 && "Expected only one function.");
    assert(After.getData().size() == 1 && "Expected only one function.");
    handleFunctionCompare(Name, Prefix, PassID, false,
                          Before.getData().begin()->getValue(),
                          After.getData().begin()->getValue());
    return;
  }

  ChangedIRData::report(
      Before, After, [&](const ChangedFuncData *B, const ChangedFuncData *A) {
        ChangedFuncData Missing;
        if (!B)
          B = &Missing;
        else if (!A)
          A = &Missing;
        assert(B != &Missing && A != &Missing &&
               "Both functions cannot be missing.");
        handleFunctionCompare(Name, Prefix, PassID, true, *B, *A);
      });
}

void ChangedIRComparer::analyzeIR(Any IR, ChangedIRData &Data) {
  if (const Module *M = getModuleForComparison(IR)) {
    // Create data for each existing/interesting function in the module.
    for (const Function &F : *M)
      generateFunctionData(Data, F);
    return;
  }

  const Function *F = nullptr;
  if (any_isa<const Function *>(IR))
    F = any_cast<const Function *>(IR);
  else {
    assert(any_isa<const Loop *>(IR) && "Unknown IR unit.");
    const Loop *L = any_cast<const Loop *>(IR);
    F = L->getHeader()->getParent();
  }
  assert(F && "Unknown IR unit.");
  generateFunctionData(Data, *F);
}

const Module *ChangedIRComparer::getModuleForComparison(Any IR) {
  if (any_isa<const Module *>(IR))
    return any_cast<const Module *>(IR);
  if (any_isa<const LazyCallGraph::SCC *>(IR))
    return any_cast<const LazyCallGraph::SCC *>(IR)
        ->begin()
        ->getFunction()
        .getParent();
  return nullptr;
}

bool ChangedIRComparer::generateFunctionData(ChangedIRData &Data,
                                             const Function &F) {
  if (!F.isDeclaration() && isFunctionInPrintList(F.getName())) {
    ChangedFuncData CFD;
    for (const auto &B : F) {
      CFD.getOrder().emplace_back(B.getName());
      CFD.getData().insert({B.getName(), B});
    }
    Data.getOrder().emplace_back(F.getName());
    Data.getData().insert({F.getName(), CFD});
    return true;
  }
  return false;
}

PrintIRInstrumentation::~PrintIRInstrumentation() {
  assert(ModuleDescStack.empty() && "ModuleDescStack is not empty at exit");
}

void PrintIRInstrumentation::pushModuleDesc(StringRef PassID, Any IR) {
  assert(StoreModuleDesc);
  const Module *M = unwrapModule(IR);
  ModuleDescStack.emplace_back(M, getIRName(IR), PassID);
}

PrintIRInstrumentation::PrintModuleDesc
PrintIRInstrumentation::popModuleDesc(StringRef PassID) {
  assert(!ModuleDescStack.empty() && "empty ModuleDescStack");
  PrintModuleDesc ModuleDesc = ModuleDescStack.pop_back_val();
  assert(std::get<2>(ModuleDesc).equals(PassID) && "malformed ModuleDescStack");
  return ModuleDesc;
}

void PrintIRInstrumentation::printBeforePass(StringRef PassID, Any IR) {
  if (isIgnored(PassID))
    return;

  // Saving Module for AfterPassInvalidated operations.
  // Note: here we rely on a fact that we do not change modules while
  // traversing the pipeline, so the latest captured module is good
  // for all print operations that has not happen yet.
  if (StoreModuleDesc && shouldPrintAfterPass(PassID))
    pushModuleDesc(PassID, IR);

  if (!shouldPrintBeforePass(PassID))
    return;

  if (!shouldPrintIR(IR))
    return;

  dbgs() << "*** IR Dump Before " << PassID << " on " << getIRName(IR)
         << " ***\n";
  unwrapAndPrint(dbgs(), IR);
}

void PrintIRInstrumentation::printAfterPass(StringRef PassID, Any IR) {
  if (isIgnored(PassID))
    return;

  if (!shouldPrintAfterPass(PassID))
    return;

  if (StoreModuleDesc) {
    const Module *M;
    std::string IRName;
    StringRef StoredPassID;
    std::tie(M, IRName, StoredPassID) = popModuleDesc(PassID);
    assert(StoredPassID == PassID && "mismatched PassID");
  }

  if (!shouldPrintIR(IR))
    return;

  dbgs() << "*** IR Dump After " << PassID << " on " << getIRName(IR)
         << " ***\n";
  unwrapAndPrint(dbgs(), IR);
}

void PrintIRInstrumentation::printAfterPassInvalidated(StringRef PassID) {
  StringRef PassName = PIC->getPassNameForClassName(PassID);
  if (!StoreModuleDesc || !shouldPrintAfterPass(PassName))
    return;

  if (isIgnored(PassID))
    return;

  const Module *M;
  std::string IRName;
  StringRef StoredPassID;
  std::tie(M, IRName, StoredPassID) = popModuleDesc(PassID);
  assert(StoredPassID == PassID && "mismatched PassID");
  // Additional filtering (e.g. -filter-print-func) can lead to module
  // printing being skipped.
  if (!M)
    return;

  SmallString<20> Banner =
      formatv("*** IR Dump After {0} on {1} (invalidated) ***", PassID, IRName);
  dbgs() << Banner << "\n";
  printIR(dbgs(), M);
}

bool PrintIRInstrumentation::shouldPrintBeforePass(StringRef PassID) {
  if (shouldPrintBeforeAll())
    return true;

  StringRef PassName = PIC->getPassNameForClassName(PassID);
  return llvm::is_contained(printBeforePasses(), PassName);
}

bool PrintIRInstrumentation::shouldPrintAfterPass(StringRef PassID) {
  if (shouldPrintAfterAll())
    return true;

  StringRef PassName = PIC->getPassNameForClassName(PassID);
  return llvm::is_contained(printAfterPasses(), PassName);
}

void PrintIRInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  this->PIC = &PIC;

  // BeforePass callback is not just for printing, it also saves a Module
  // for later use in AfterPassInvalidated.
  StoreModuleDesc = forcePrintModuleIR() && shouldPrintAfterSomePass();
  if (shouldPrintBeforeSomePass() || StoreModuleDesc)
    PIC.registerBeforeNonSkippedPassCallback(
        [this](StringRef P, Any IR) { this->printBeforePass(P, IR); });

  if (shouldPrintAfterSomePass()) {
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
  PIC.registerShouldRunOptionalPassCallback(
      [this](StringRef P, Any IR) { return this->shouldRun(P, IR); });
}

bool OptNoneInstrumentation::shouldRun(StringRef PassID, Any IR) {
  const Function *F = nullptr;
  if (any_isa<const Function *>(IR)) {
    F = any_cast<const Function *>(IR);
  } else if (any_isa<const Loop *>(IR)) {
    F = any_cast<const Loop *>(IR)->getHeader()->getParent();
  }
  bool ShouldRun = !(F && F->hasOptNone());
  if (!ShouldRun && DebugLogging) {
    errs() << "Skipping pass " << PassID << " on " << F->getName()
           << " due to optnone attribute\n";
  }
  return ShouldRun;
}

void OptBisectInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  if (!OptBisector->isEnabled())
    return;
  PIC.registerShouldRunOptionalPassCallback([](StringRef PassID, Any IR) {
    return isIgnored(PassID) || OptBisector->checkPass(PassID, getIRName(IR));
  });
}

raw_ostream &PrintPassInstrumentation::print() {
  if (Opts.Indent) {
    assert(Indent >= 0);
    dbgs().indent(Indent);
  }
  return dbgs();
}

void PrintPassInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  if (!Enabled)
    return;

  std::vector<StringRef> SpecialPasses;
  if (!Opts.Verbose) {
    SpecialPasses.emplace_back("PassManager");
    SpecialPasses.emplace_back("PassAdaptor");
  }

  PIC.registerBeforeSkippedPassCallback(
      [this, SpecialPasses](StringRef PassID, Any IR) {
        assert(!isSpecialPass(PassID, SpecialPasses) &&
               "Unexpectedly skipping special pass");

        print() << "Skipping pass: " << PassID << " on " << getIRName(IR)
                << "\n";
      });
  PIC.registerBeforeNonSkippedPassCallback([this, SpecialPasses](
                                               StringRef PassID, Any IR) {
    if (isSpecialPass(PassID, SpecialPasses))
      return;

    print() << "Running pass: " << PassID << " on " << getIRName(IR) << "\n";
    Indent += 2;
  });
  PIC.registerAfterPassCallback(
      [this, SpecialPasses](StringRef PassID, Any IR,
                            const PreservedAnalyses &) {
        if (isSpecialPass(PassID, SpecialPasses))
          return;

        Indent -= 2;
      });
  PIC.registerAfterPassInvalidatedCallback(
      [this, SpecialPasses](StringRef PassID, Any IR) {
        if (isSpecialPass(PassID, SpecialPasses))
          return;

        Indent -= 2;
      });

  if (!Opts.SkipAnalyses) {
    PIC.registerBeforeAnalysisCallback([this](StringRef PassID, Any IR) {
      print() << "Running analysis: " << PassID << " on " << getIRName(IR)
              << "\n";
      Indent += 2;
    });
    PIC.registerAfterAnalysisCallback(
        [this](StringRef PassID, Any IR) { Indent -= 2; });
    PIC.registerAnalysisInvalidatedCallback([this](StringRef PassID, Any IR) {
      print() << "Invalidating analysis: " << PassID << " on " << getIRName(IR)
              << "\n";
    });
    PIC.registerAnalysesClearedCallback([this](StringRef IRName) {
      print() << "Clearing all analysis results for: " << IRName << "\n";
    });
  }
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

  if (BB->isEntryBlock()) {
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

// PreservedCFGCheckerInstrumentation uses PreservedCFGCheckerAnalysis to check
// passes, that reported they kept CFG analyses up-to-date, did not actually
// change CFG. This check is done as follows. Before every functional pass in
// BeforeNonSkippedPassCallback a CFG snapshot (an instance of
// PreservedCFGCheckerInstrumentation::CFG) is requested from
// FunctionAnalysisManager as a result of PreservedCFGCheckerAnalysis. When the
// functional pass finishes and reports that CFGAnalyses or AllAnalyses are
// up-to-date then the cached result of PreservedCFGCheckerAnalysis (if
// available) is checked to be equal to a freshly created CFG snapshot.
struct PreservedCFGCheckerAnalysis
    : public AnalysisInfoMixin<PreservedCFGCheckerAnalysis> {
  friend AnalysisInfoMixin<PreservedCFGCheckerAnalysis>;

  static AnalysisKey Key;

public:
  /// Provide the result type for this analysis pass.
  using Result = PreservedCFGCheckerInstrumentation::CFG;

  /// Run the analysis pass over a function and produce CFG.
  Result run(Function &F, FunctionAnalysisManager &FAM) {
    return Result(&F, /* TrackBBLifetime */ true);
  }
};

AnalysisKey PreservedCFGCheckerAnalysis::Key;

bool PreservedCFGCheckerInstrumentation::CFG::invalidate(
    Function &F, const PreservedAnalyses &PA,
    FunctionAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<PreservedCFGCheckerAnalysis>();
  return !(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>() ||
           PAC.preservedSet<CFGAnalyses>());
}

void PreservedCFGCheckerInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC, FunctionAnalysisManager &FAM) {
  if (!VerifyPreservedCFG)
    return;

  FAM.registerPass([&] { return PreservedCFGCheckerAnalysis(); });

  auto checkCFG = [](StringRef Pass, StringRef FuncName, const CFG &GraphBefore,
                     const CFG &GraphAfter) {
    if (GraphAfter == GraphBefore)
      return;

    dbgs() << "Error: " << Pass
           << " does not invalidate CFG analyses but CFG changes detected in "
              "function @"
           << FuncName << ":\n";
    CFG::printDiff(dbgs(), GraphBefore, GraphAfter);
    report_fatal_error(Twine("CFG unexpectedly changed by ", Pass));
  };

  PIC.registerBeforeNonSkippedPassCallback(
      [this, &FAM](StringRef P, Any IR) {
        assert(&PassStack.emplace_back(P));
        (void)this;
        if (!any_isa<const Function *>(IR))
          return;

        const auto *F = any_cast<const Function *>(IR);
        // Make sure a fresh CFG snapshot is available before the pass.
        FAM.getResult<PreservedCFGCheckerAnalysis>(*const_cast<Function *>(F));
      });

  PIC.registerAfterPassInvalidatedCallback(
      [this](StringRef P, const PreservedAnalyses &PassPA) {
        assert(PassStack.pop_back_val() == P &&
               "Before and After callbacks must correspond");
        (void)this;
      });

  PIC.registerAfterPassCallback([this, &FAM,
                                 checkCFG](StringRef P, Any IR,
                                           const PreservedAnalyses &PassPA) {
    assert(PassStack.pop_back_val() == P &&
           "Before and After callbacks must correspond");
    (void)this;

    if (!any_isa<const Function *>(IR))
      return;

    if (!PassPA.allAnalysesInSetPreserved<CFGAnalyses>() &&
        !PassPA.allAnalysesInSetPreserved<AllAnalysesOn<Function>>())
      return;

    const auto *F = any_cast<const Function *>(IR);
    if (auto *GraphBefore = FAM.getCachedResult<PreservedCFGCheckerAnalysis>(
            *const_cast<Function *>(F)))
      checkCFG(P, F->getName(), *GraphBefore,
               CFG(F, /* TrackBBLifetime */ false));
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

InLineChangePrinter::~InLineChangePrinter() {}

void InLineChangePrinter::generateIRRepresentation(Any IR, StringRef PassID,
                                                   ChangedIRData &D) {
  ChangedIRComparer::analyzeIR(IR, D);
}

void InLineChangePrinter::handleAfter(StringRef PassID, std::string &Name,
                                      const ChangedIRData &Before,
                                      const ChangedIRData &After, Any IR) {
  SmallString<20> Banner =
      formatv("*** IR Dump After {0} on {1} ***\n", PassID, Name);
  Out << Banner;
  ChangedIRComparer(Out, Before, After, UseColour)
      .compare(IR, "", PassID, Name);
  Out << "\n";
}

bool InLineChangePrinter::same(const ChangedIRData &D1,
                               const ChangedIRData &D2) {
  return D1 == D2;
}

void ChangedIRComparer::handleFunctionCompare(StringRef Name, StringRef Prefix,
                                              StringRef PassID, bool InModule,
                                              const ChangedFuncData &Before,
                                              const ChangedFuncData &After) {
  // Print a banner when this is being shown in the context of a module
  if (InModule)
    Out << "\n*** IR for function " << Name << " ***\n";

  ChangedFuncData::report(
      Before, After, [&](const ChangedBlockData *B, const ChangedBlockData *A) {
        StringRef BStr = B ? B->getBody() : "\n";
        StringRef AStr = A ? A->getBody() : "\n";
        const std::string Removed =
            UseColour ? "\033[31m-%l\033[0m\n" : "-%l\n";
        const std::string Added = UseColour ? "\033[32m+%l\033[0m\n" : "+%l\n";
        const std::string NoChange = " %l\n";
        Out << doSystemDiff(BStr, AStr, Removed, Added, NoChange);
      });
}

void InLineChangePrinter::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (PrintChanged == ChangePrinter::PrintChangedDiffVerbose ||
      PrintChanged == ChangePrinter::PrintChangedDiffQuiet ||
      PrintChanged == ChangePrinter::PrintChangedColourDiffVerbose ||
      PrintChanged == ChangePrinter::PrintChangedColourDiffQuiet)
    TextChangeReporter<ChangedIRData>::registerRequiredCallbacks(PIC);
}

StandardInstrumentations::StandardInstrumentations(
    bool DebugLogging, bool VerifyEach, PrintPassOptions PrintPassOpts)
    : PrintPass(DebugLogging, PrintPassOpts), OptNone(DebugLogging),
      PrintChangedIR(PrintChanged == ChangePrinter::PrintChangedVerbose),
      PrintChangedDiff(
          PrintChanged == ChangePrinter::PrintChangedDiffVerbose ||
              PrintChanged == ChangePrinter::PrintChangedColourDiffVerbose,
          PrintChanged == ChangePrinter::PrintChangedColourDiffVerbose ||
              PrintChanged == ChangePrinter::PrintChangedColourDiffQuiet),
      Verify(DebugLogging), VerifyEach(VerifyEach) {}

void StandardInstrumentations::registerCallbacks(
    PassInstrumentationCallbacks &PIC, FunctionAnalysisManager *FAM) {
  PrintIR.registerCallbacks(PIC);
  PrintPass.registerCallbacks(PIC);
  TimePasses.registerCallbacks(PIC);
  OptNone.registerCallbacks(PIC);
  OptBisect.registerCallbacks(PIC);
  if (FAM)
    PreservedCFGChecker.registerCallbacks(PIC, *FAM);
  PrintChangedIR.registerCallbacks(PIC);
  PseudoProbeVerification.registerCallbacks(PIC);
  if (VerifyEach)
    Verify.registerCallbacks(PIC);
  PrintChangedDiff.registerCallbacks(PIC);
}

namespace llvm {

template class ChangeReporter<std::string>;
template class TextChangeReporter<std::string>;

template class ChangeReporter<ChangedIRData>;
template class TextChangeReporter<ChangedIRData>;

} // namespace llvm
