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
#include "llvm/IR/Constants.h"
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
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace llvm;

cl::opt<bool> PreservedCFGCheckerInstrumentation::VerifyPreservedCFG(
    "verify-cfg-preserved", cl::Hidden,
#ifdef NDEBUG
    cl::init(false)
#else
    cl::init(true)
#endif
    );

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
  PrintChangedColourDiffQuiet,
  PrintChangedDotCfgVerbose,
  PrintChangedDotCfgQuiet
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
        clEnumValN(ChangePrinter::PrintChangedDotCfgVerbose, "dot-cfg",
                   "Create a website with graphical changes"),
        clEnumValN(ChangePrinter::PrintChangedDotCfgQuiet, "dot-cfg-quiet",
                   "Create a website with graphical changes in quiet mode"),
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

// An option for specifying the dot used by
// print-changed=[dot-cfg | dot-cfg-quiet]
static cl::opt<std::string>
    DotBinary("print-changed-dot-path", cl::Hidden, cl::init("dot"),
              cl::desc("system dot used by change reporters"));

// An option that determines the colour used for elements that are only
// in the before part.  Must be a colour named in appendix J of
// https://graphviz.org/pdf/dotguide.pdf
cl::opt<std::string>
    BeforeColour("dot-cfg-before-color",
                 cl::desc("Color for dot-cfg before elements."), cl::Hidden,
                 cl::init("red"));
// An option that determines the colour used for elements that are only
// in the after part.  Must be a colour named in appendix J of
// https://graphviz.org/pdf/dotguide.pdf
cl::opt<std::string> AfterColour("dot-cfg-after-color",
                                 cl::desc("Color for dot-cfg after elements."),
                                 cl::Hidden, cl::init("forestgreen"));
// An option that determines the colour used for elements that are in both
// the before and after parts.  Must be a colour named in appendix J of
// https://graphviz.org/pdf/dotguide.pdf
cl::opt<std::string>
    CommonColour("dot-cfg-common-color",
                 cl::desc("Color for dot-cfg common elements."), cl::Hidden,
                 cl::init("black"));

// An option that determines where the generated website file (named
// passes.html) and the associated pdf files (named diff_*.pdf) are saved.
static cl::opt<std::string> DotCfgDir(
    "dot-cfg-dir",
    cl::desc("Generate dot files into specified directory for changed IRs"),
    cl::Hidden, cl::init("./"));

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

  StringRef Args[] = {DiffBinary, "-w", "-d",        OLF,
                      NLF,        ULF,  FileName[0], FileName[1]};
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
  for (const std::string &I : FileName) {
    std::error_code EC = sys::fs::remove(I);
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

void printIR(raw_ostream &OS, const Module *M) {
  if (isFunctionInPrintList("*") || forcePrintModuleIR()) {
    M->print(OS, nullptr);
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
void unwrapAndPrint(raw_ostream &OS, Any IR) {
  if (!shouldPrintIR(IR))
    return;

  if (forcePrintModuleIR()) {
    auto *M = unwrapModule(IR);
    assert(M && "should have unwrapped module");
    printIR(OS, M);
    return;
  }

  if (any_isa<const Module *>(IR)) {
    const Module *M = any_cast<const Module *>(IR);
    printIR(OS, M);
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

std::string makeHTMLReady(StringRef SR) {
  std::string S;
  while (true) {
    StringRef Clean =
        SR.take_until([](char C) { return C == '<' || C == '>'; });
    S.append(Clean.str());
    SR = SR.drop_front(Clean.size());
    if (SR.size() == 0)
      return S;
    S.append(SR[0] == '<' ? "&lt;" : "&gt;");
    SR = SR.drop_front();
  }
  llvm_unreachable("problems converting string to HTML");
}

// Return the module when that is the appropriate level of comparison for \p IR.
const Module *getModuleForComparison(Any IR) {
  if (any_isa<const Module *>(IR))
    return any_cast<const Module *>(IR);
  if (any_isa<const LazyCallGraph::SCC *>(IR))
    return any_cast<const LazyCallGraph::SCC *>(IR)
        ->begin()
        ->getFunction()
        .getParent();
  return nullptr;
}

bool isInterestingFunction(const Function &F) {
  return isFunctionInPrintList(F.getName());
}

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

template <typename T> ChangeReporter<T>::~ChangeReporter<T>() {
  assert(BeforeStack.empty() && "Problem with Change Printer stack.");
}

template <typename T>
void ChangeReporter<T>::saveIRBeforePass(Any IR, StringRef PassID) {
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
  T &Data = BeforeStack.back();
  generateIRRepresentation(IR, PassID, Data);
}

template <typename T>
void ChangeReporter<T>::handleIRAfterPass(Any IR, StringRef PassID) {
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
    T &Before = BeforeStack.back();
    // Create the after rep
    T After;
    generateIRRepresentation(IR, PassID, After);

    // Was there a change in IR?
    if (Before == After) {
      if (VerboseMode)
        omitAfter(PassID, Name);
    } else
      handleAfter(PassID, Name, Before, After, IR);
  }
  BeforeStack.pop_back();
}

template <typename T>
void ChangeReporter<T>::handleInvalidatedPass(StringRef PassID) {
  assert(!BeforeStack.empty() && "Unexpected empty stack encountered.");

  // Always flag it as invalidated as we cannot determine when
  // a pass for a filtered function is invalidated since we do not
  // get the IR in the call.  Also, the output is just alternate
  // forms of the banner anyway.
  if (VerboseMode)
    handleInvalidated(PassID);
  BeforeStack.pop_back();
}

template <typename T>
void ChangeReporter<T>::registerRequiredCallbacks(
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

template <typename T>
TextChangeReporter<T>::TextChangeReporter(bool Verbose)
    : ChangeReporter<T>(Verbose), Out(dbgs()) {}

template <typename T> void TextChangeReporter<T>::handleInitialIR(Any IR) {
  // Always print the module.
  // Unwrap and print directly to avoid filtering problems in general routines.
  auto *M = unwrapModule(IR, /*Force=*/true);
  assert(M && "Expected module to be unwrapped when forced.");
  Out << "*** IR Dump At Start ***\n";
  M->print(Out, nullptr);
}

template <typename T>
void TextChangeReporter<T>::omitAfter(StringRef PassID, std::string &Name) {
  Out << formatv("*** IR Dump After {0} on {1} omitted because no change ***\n",
                 PassID, Name);
}

template <typename T>
void TextChangeReporter<T>::handleInvalidated(StringRef PassID) {
  Out << formatv("*** IR Pass {0} invalidated ***\n", PassID);
}

template <typename T>
void TextChangeReporter<T>::handleFiltered(StringRef PassID,
                                           std::string &Name) {
  SmallString<20> Banner =
      formatv("*** IR Dump After {0} on {1} filtered out ***\n", PassID, Name);
  Out << Banner;
}

template <typename T>
void TextChangeReporter<T>::handleIgnored(StringRef PassID, std::string &Name) {
  Out << formatv("*** IR Pass {0} on {1} ignored ***\n", PassID, Name);
}

IRChangedPrinter::~IRChangedPrinter() = default;

void IRChangedPrinter::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (PrintChanged == ChangePrinter::PrintChangedVerbose ||
      PrintChanged == ChangePrinter::PrintChangedQuiet)
    TextChangeReporter<std::string>::registerRequiredCallbacks(PIC);
}

void IRChangedPrinter::generateIRRepresentation(Any IR, StringRef PassID,
                                                std::string &Output) {
  raw_string_ostream OS(Output);
  unwrapAndPrint(OS, IR);
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

template <typename T>
void OrderedChangedData<T>::report(
    const OrderedChangedData &Before, const OrderedChangedData &After,
    function_ref<void(const T *, const T *)> HandlePair) {
  const auto &BFD = Before.getData();
  const auto &AFD = After.getData();
  std::vector<std::string>::const_iterator BI = Before.getOrder().begin();
  std::vector<std::string>::const_iterator BE = Before.getOrder().end();
  std::vector<std::string>::const_iterator AI = After.getOrder().begin();
  std::vector<std::string>::const_iterator AE = After.getOrder().end();

  auto HandlePotentiallyRemovedData = [&](std::string S) {
    // The order in LLVM may have changed so check if still exists.
    if (!AFD.count(S)) {
      // This has been removed.
      HandlePair(&BFD.find(*BI)->getValue(), nullptr);
    }
  };
  auto HandleNewData = [&](std::vector<const T *> &Q) {
    // Print out any queued up new sections
    for (const T *NBI : Q)
      HandlePair(nullptr, NBI);
    Q.clear();
  };

  // Print out the data in the after order, with before ones interspersed
  // appropriately (ie, somewhere near where they were in the before list).
  // Start at the beginning of both lists.  Loop through the
  // after list.  If an element is common, then advance in the before list
  // reporting the removed ones until the common one is reached.  Report any
  // queued up new ones and then report the common one.  If an element is not
  // common, then enqueue it for reporting.  When the after list is exhausted,
  // loop through the before list, reporting any removed ones.  Finally,
  // report the rest of the enqueued new ones.
  std::vector<const T *> NewDataQueue;
  while (AI != AE) {
    if (!BFD.count(*AI)) {
      // This section is new so place it in the queue.  This will cause it
      // to be reported after deleted sections.
      NewDataQueue.emplace_back(&AFD.find(*AI)->getValue());
      ++AI;
      continue;
    }
    // This section is in both; advance and print out any before-only
    // until we get to it.
    while (*BI != *AI) {
      HandlePotentiallyRemovedData(*BI);
      ++BI;
    }
    // Report any new sections that were queued up and waiting.
    HandleNewData(NewDataQueue);

    const T &AData = AFD.find(*AI)->getValue();
    const T &BData = BFD.find(*AI)->getValue();
    HandlePair(&BData, &AData);
    ++BI;
    ++AI;
  }

  // Check any remaining before sections to see if they have been removed
  while (BI != BE) {
    HandlePotentiallyRemovedData(*BI);
    ++BI;
  }

  HandleNewData(NewDataQueue);
}

template <typename T>
void IRComparer<T>::compare(
    bool CompareModule,
    std::function<void(bool InModule, unsigned Minor,
                       const FuncDataT<T> &Before, const FuncDataT<T> &After)>
        CompareFunc) {
  if (!CompareModule) {
    // Just handle the single function.
    assert(Before.getData().size() == 1 && After.getData().size() == 1 &&
           "Expected only one function.");
    CompareFunc(false, 0, Before.getData().begin()->getValue(),
                After.getData().begin()->getValue());
    return;
  }

  unsigned Minor = 0;
  FuncDataT<T> Missing("");
  IRDataT<T>::report(Before, After,
                     [&](const FuncDataT<T> *B, const FuncDataT<T> *A) {
                       assert((B || A) && "Both functions cannot be missing.");
                       if (!B)
                         B = &Missing;
                       else if (!A)
                         A = &Missing;
                       CompareFunc(true, Minor++, *B, *A);
                     });
}

template <typename T> void IRComparer<T>::analyzeIR(Any IR, IRDataT<T> &Data) {
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

template <typename T>
bool IRComparer<T>::generateFunctionData(IRDataT<T> &Data, const Function &F) {
  if (!F.isDeclaration() && isFunctionInPrintList(F.getName())) {
    FuncDataT<T> FD(F.getEntryBlock().getName().str());
    for (const auto &B : F) {
      FD.getOrder().emplace_back(B.getName());
      FD.getData().insert({B.getName(), B});
    }
    Data.getOrder().emplace_back(F.getName());
    Data.getData().insert({F.getName(), FD});
    return true;
  }
  return false;
}

PrintIRInstrumentation::~PrintIRInstrumentation() {
  assert(ModuleDescStack.empty() && "ModuleDescStack is not empty at exit");
}

void PrintIRInstrumentation::pushModuleDesc(StringRef PassID, Any IR) {
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
  if (shouldPrintAfterPass(PassID))
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

  const Module *M;
  std::string IRName;
  StringRef StoredPassID;
  std::tie(M, IRName, StoredPassID) = popModuleDesc(PassID);
  assert(StoredPassID == PassID && "mismatched PassID");

  if (!shouldPrintIR(IR))
    return;

  dbgs() << "*** IR Dump After " << PassID << " on " << IRName << " ***\n";
  unwrapAndPrint(dbgs(), IR);
}

void PrintIRInstrumentation::printAfterPassInvalidated(StringRef PassID) {
  StringRef PassName = PIC->getPassNameForClassName(PassID);
  if (!shouldPrintAfterPass(PassName))
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
  return is_contained(printBeforePasses(), PassName);
}

bool PrintIRInstrumentation::shouldPrintAfterPass(StringRef PassID) {
  if (shouldPrintAfterAll())
    return true;

  StringRef PassName = PIC->getPassNameForClassName(PassID);
  return is_contained(printAfterPasses(), PassName);
}

void PrintIRInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  this->PIC = &PIC;

  // BeforePass callback is not just for printing, it also saves a Module
  // for later use in AfterPassInvalidated.
  if (shouldPrintBeforeSomePass() || shouldPrintAfterSomePass())
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

  PIC.registerBeforeSkippedPassCallback([this, SpecialPasses](StringRef PassID,
                                                              Any IR) {
    assert(!isSpecialPass(PassID, SpecialPasses) &&
           "Unexpectedly skipping special pass");

    print() << "Skipping pass: " << PassID << " on " << getIRName(IR) << "\n";
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

  PIC.registerBeforeNonSkippedPassCallback([this, &FAM](StringRef P, Any IR) {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(&PassStack.emplace_back(P));
#endif
    (void)this;
    if (!any_isa<const Function *>(IR))
      return;

    const auto *F = any_cast<const Function *>(IR);
    // Make sure a fresh CFG snapshot is available before the pass.
    FAM.getResult<PreservedCFGCheckerAnalysis>(*const_cast<Function *>(F));
  });

  PIC.registerAfterPassInvalidatedCallback(
      [this](StringRef P, const PreservedAnalyses &PassPA) {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
        assert(PassStack.pop_back_val() == P &&
               "Before and After callbacks must correspond");
#endif
        (void)this;
      });

  PIC.registerAfterPassCallback([this, &FAM,
                                 checkCFG](StringRef P, Any IR,
                                           const PreservedAnalyses &PassPA) {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(PassStack.pop_back_val() == P &&
           "Before and After callbacks must correspond");
#endif
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

InLineChangePrinter::~InLineChangePrinter() = default;

void InLineChangePrinter::generateIRRepresentation(Any IR, StringRef PassID,
                                                   IRDataT<EmptyData> &D) {
  IRComparer<EmptyData>::analyzeIR(IR, D);
}

void InLineChangePrinter::handleAfter(StringRef PassID, std::string &Name,
                                      const IRDataT<EmptyData> &Before,
                                      const IRDataT<EmptyData> &After, Any IR) {
  SmallString<20> Banner =
      formatv("*** IR Dump After {0} on {1} ***\n", PassID, Name);
  Out << Banner;
  IRComparer<EmptyData>(Before, After)
      .compare(getModuleForComparison(IR),
               [&](bool InModule, unsigned Minor,
                   const FuncDataT<EmptyData> &Before,
                   const FuncDataT<EmptyData> &After) -> void {
                 handleFunctionCompare(Name, "", PassID, " on ", InModule,
                                       Minor, Before, After);
               });
  Out << "\n";
}

void InLineChangePrinter::handleFunctionCompare(
    StringRef Name, StringRef Prefix, StringRef PassID, StringRef Divider,
    bool InModule, unsigned Minor, const FuncDataT<EmptyData> &Before,
    const FuncDataT<EmptyData> &After) {
  // Print a banner when this is being shown in the context of a module
  if (InModule)
    Out << "\n*** IR for function " << Name << " ***\n";

  FuncDataT<EmptyData>::report(
      Before, After,
      [&](const BlockDataT<EmptyData> *B, const BlockDataT<EmptyData> *A) {
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
    TextChangeReporter<IRDataT<EmptyData>>::registerRequiredCallbacks(PIC);
}

namespace {

class DisplayNode;
class DotCfgDiffDisplayGraph;

// Base class for a node or edge in the dot-cfg-changes graph.
class DisplayElement {
public:
  // Is this in before, after, or both?
  StringRef getColour() const { return Colour; }

protected:
  DisplayElement(StringRef Colour) : Colour(Colour) {}
  const StringRef Colour;
};

// An edge representing a transition between basic blocks in the
// dot-cfg-changes graph.
class DisplayEdge : public DisplayElement {
public:
  DisplayEdge(std::string Value, DisplayNode &Node, StringRef Colour)
      : DisplayElement(Colour), Value(Value), Node(Node) {}
  // The value on which the transition is made.
  std::string getValue() const { return Value; }
  // The node (representing a basic block) reached by this transition.
  const DisplayNode &getDestinationNode() const { return Node; }

protected:
  std::string Value;
  const DisplayNode &Node;
};

// A node in the dot-cfg-changes graph which represents a basic block.
class DisplayNode : public DisplayElement {
public:
  // \p C is the content for the node, \p T indicates the colour for the
  // outline of the node
  DisplayNode(std::string Content, StringRef Colour)
      : DisplayElement(Colour), Content(Content) {}

  // Iterator to the child nodes.  Required by GraphWriter.
  using ChildIterator = std::unordered_set<DisplayNode *>::const_iterator;
  ChildIterator children_begin() const { return Children.cbegin(); }
  ChildIterator children_end() const { return Children.cend(); }

  // Iterator for the edges.  Required by GraphWriter.
  using EdgeIterator = std::vector<DisplayEdge *>::const_iterator;
  EdgeIterator edges_begin() const { return EdgePtrs.cbegin(); }
  EdgeIterator edges_end() const { return EdgePtrs.cend(); }

  // Create an edge to \p Node on value \p Value, with colour \p Colour.
  void createEdge(StringRef Value, DisplayNode &Node, StringRef Colour);

  // Return the content of this node.
  std::string getContent() const { return Content; }

  // Return the edge to node \p S.
  const DisplayEdge &getEdge(const DisplayNode &To) const {
    assert(EdgeMap.find(&To) != EdgeMap.end() && "Expected to find edge.");
    return *EdgeMap.find(&To)->second;
  }

  // Return the value for the transition to basic block \p S.
  // Required by GraphWriter.
  std::string getEdgeSourceLabel(const DisplayNode &Sink) const {
    return getEdge(Sink).getValue();
  }

  void createEdgeMap();

protected:
  const std::string Content;

  // Place to collect all of the edges.  Once they are all in the vector,
  // the vector will not reallocate so then we can use pointers to them,
  // which are required by the graph writing routines.
  std::vector<DisplayEdge> Edges;

  std::vector<DisplayEdge *> EdgePtrs;
  std::unordered_set<DisplayNode *> Children;
  std::unordered_map<const DisplayNode *, const DisplayEdge *> EdgeMap;

  // Safeguard adding of edges.
  bool AllEdgesCreated = false;
};

// Class representing a difference display (corresponds to a pdf file).
class DotCfgDiffDisplayGraph {
public:
  DotCfgDiffDisplayGraph(std::string Name) : GraphName(Name) {}

  // Generate the file into \p DotFile.
  void generateDotFile(StringRef DotFile);

  // Iterator to the nodes.  Required by GraphWriter.
  using NodeIterator = std::vector<DisplayNode *>::const_iterator;
  NodeIterator nodes_begin() const {
    assert(NodeGenerationComplete && "Unexpected children iterator creation");
    return NodePtrs.cbegin();
  }
  NodeIterator nodes_end() const {
    assert(NodeGenerationComplete && "Unexpected children iterator creation");
    return NodePtrs.cend();
  }

  // Record the index of the entry node.  At this point, we can build up
  // vectors of pointers that are required by the graph routines.
  void setEntryNode(unsigned N) {
    // At this point, there will be no new nodes.
    assert(!NodeGenerationComplete && "Unexpected node creation");
    NodeGenerationComplete = true;
    for (auto &N : Nodes)
      NodePtrs.emplace_back(&N);

    EntryNode = NodePtrs[N];
  }

  // Create a node.
  void createNode(std::string C, StringRef Colour) {
    assert(!NodeGenerationComplete && "Unexpected node creation");
    Nodes.emplace_back(C, Colour);
  }
  // Return the node at index \p N to avoid problems with vectors reallocating.
  DisplayNode &getNode(unsigned N) {
    assert(N < Nodes.size() && "Node is out of bounds");
    return Nodes[N];
  }
  unsigned size() const {
    assert(NodeGenerationComplete && "Unexpected children iterator creation");
    return Nodes.size();
  }

  // Return the name of the graph.  Required by GraphWriter.
  std::string getGraphName() const { return GraphName; }

  // Return the string representing the differences for basic block \p Node.
  // Required by GraphWriter.
  std::string getNodeLabel(const DisplayNode &Node) const {
    return Node.getContent();
  }

  // Return a string with colour information for Dot.  Required by GraphWriter.
  std::string getNodeAttributes(const DisplayNode &Node) const {
    return attribute(Node.getColour());
  }

  // Return a string with colour information for Dot.  Required by GraphWriter.
  std::string getEdgeColorAttr(const DisplayNode &From,
                               const DisplayNode &To) const {
    return attribute(From.getEdge(To).getColour());
  }

  // Get the starting basic block.  Required by GraphWriter.
  DisplayNode *getEntryNode() const {
    assert(NodeGenerationComplete && "Unexpected children iterator creation");
    return EntryNode;
  }

protected:
  // Return the string containing the colour to use as a Dot attribute.
  std::string attribute(StringRef Colour) const {
    return "color=" + Colour.str();
  }

  bool NodeGenerationComplete = false;
  const std::string GraphName;
  std::vector<DisplayNode> Nodes;
  std::vector<DisplayNode *> NodePtrs;
  DisplayNode *EntryNode = nullptr;
};

void DisplayNode::createEdge(StringRef Value, DisplayNode &Node,
                             StringRef Colour) {
  assert(!AllEdgesCreated && "Expected to be able to still create edges.");
  Edges.emplace_back(Value.str(), Node, Colour);
  Children.insert(&Node);
}

void DisplayNode::createEdgeMap() {
  // No more edges will be added so we can now use pointers to the edges
  // as the vector will not grow and reallocate.
  AllEdgesCreated = true;
  for (auto &E : Edges)
    EdgeMap.insert({&E.getDestinationNode(), &E});
}

class DotCfgDiffNode;
class DotCfgDiff;

// A class representing a basic block in the Dot difference graph.
class DotCfgDiffNode {
public:
  DotCfgDiffNode() = delete;

  // Create a node in Dot difference graph \p G representing the basic block
  // represented by \p BD with colour \p Colour (where it exists).
  DotCfgDiffNode(DotCfgDiff &G, unsigned N, const BlockDataT<DCData> &BD,
                 StringRef Colour)
      : Graph(G), N(N), Data{&BD, nullptr}, Colour(Colour) {}
  DotCfgDiffNode(const DotCfgDiffNode &DN)
      : Graph(DN.Graph), N(DN.N), Data{DN.Data[0], DN.Data[1]},
        Colour(DN.Colour), EdgesMap(DN.EdgesMap), Children(DN.Children),
        Edges(DN.Edges) {}

  unsigned getIndex() const { return N; }

  // The label of the basic block
  StringRef getLabel() const {
    assert(Data[0] && "Expected Data[0] to be set.");
    return Data[0]->getLabel();
  }
  // Return the colour for this block
  StringRef getColour() const { return Colour; }
  // Change this basic block from being only in before to being common.
  // Save the pointer to \p Other.
  void setCommon(const BlockDataT<DCData> &Other) {
    assert(!Data[1] && "Expected only one block datum");
    Data[1] = &Other;
    Colour = CommonColour;
  }
  // Add an edge to \p E of colour {\p Value, \p Colour}.
  void addEdge(unsigned E, StringRef Value, StringRef Colour) {
    // This is a new edge or it is an edge being made common.
    assert((EdgesMap.count(E) == 0 || Colour == CommonColour) &&
           "Unexpected edge count and color.");
    EdgesMap[E] = {Value.str(), Colour};
  }
  // Record the children and create edges.
  void finalize(DotCfgDiff &G);

  // Return the colour of the edge to node \p S.
  StringRef getEdgeColour(const unsigned S) const {
    assert(EdgesMap.count(S) == 1 && "Expected to find edge.");
    return EdgesMap.at(S).second;
  }

  // Return the string representing the basic block.
  std::string getBodyContent() const;

  void createDisplayEdges(DotCfgDiffDisplayGraph &Graph, unsigned DisplayNode,
                          std::map<const unsigned, unsigned> &NodeMap) const;

protected:
  DotCfgDiff &Graph;
  const unsigned N;
  const BlockDataT<DCData> *Data[2];
  StringRef Colour;
  std::map<const unsigned, std::pair<std::string, StringRef>> EdgesMap;
  std::vector<unsigned> Children;
  std::vector<unsigned> Edges;
};

// Class representing the difference graph between two functions.
class DotCfgDiff {
public:
  // \p Title is the title given to the graph.  \p EntryNodeName is the
  // entry node for the function.  \p Before and \p After are the before
  // after versions of the function, respectively.  \p Dir is the directory
  // in which to store the results.
  DotCfgDiff(StringRef Title, const FuncDataT<DCData> &Before,
             const FuncDataT<DCData> &After);

  DotCfgDiff(const DotCfgDiff &) = delete;
  DotCfgDiff &operator=(const DotCfgDiff &) = delete;

  DotCfgDiffDisplayGraph createDisplayGraph(StringRef Title,
                                            StringRef EntryNodeName);

  // Return a string consisting of the labels for the \p Source and \p Sink.
  // The combination allows distinguishing changing transitions on the
  // same value (ie, a transition went to X before and goes to Y after).
  // Required by GraphWriter.
  StringRef getEdgeSourceLabel(const unsigned &Source,
                               const unsigned &Sink) const {
    std::string S =
        getNode(Source).getLabel().str() + " " + getNode(Sink).getLabel().str();
    assert(EdgeLabels.count(S) == 1 && "Expected to find edge label.");
    return EdgeLabels.find(S)->getValue();
  }

  // Return the number of basic blocks (nodes).  Required by GraphWriter.
  unsigned size() const { return Nodes.size(); }

  const DotCfgDiffNode &getNode(unsigned N) const {
    assert(N < Nodes.size() && "Unexpected index for node reference");
    return Nodes[N];
  }

protected:
  // Return the string surrounded by HTML to make it the appropriate colour.
  std::string colourize(std::string S, StringRef Colour) const;

  void createNode(StringRef Label, const BlockDataT<DCData> &BD, StringRef C) {
    unsigned Pos = Nodes.size();
    Nodes.emplace_back(*this, Pos, BD, C);
    NodePosition.insert({Label, Pos});
  }

  // TODO Nodes should probably be a StringMap<DotCfgDiffNode> after the
  // display graph is separated out, which would remove the need for
  // NodePosition.
  std::vector<DotCfgDiffNode> Nodes;
  StringMap<unsigned> NodePosition;
  const std::string GraphName;

  StringMap<std::string> EdgeLabels;
};

std::string DotCfgDiffNode::getBodyContent() const {
  if (Colour == CommonColour) {
    assert(Data[1] && "Expected Data[1] to be set.");

    StringRef SR[2];
    for (unsigned I = 0; I < 2; ++I) {
      SR[I] = Data[I]->getBody();
      // drop initial '\n' if present
      if (SR[I][0] == '\n')
        SR[I] = SR[I].drop_front();
      // drop predecessors as they can be big and are redundant
      SR[I] = SR[I].drop_until([](char C) { return C == '\n'; }).drop_front();
    }

    SmallString<80> OldLineFormat = formatv(
        "<FONT COLOR=\"{0}\">%l</FONT><BR align=\"left\"/>", BeforeColour);
    SmallString<80> NewLineFormat = formatv(
        "<FONT COLOR=\"{0}\">%l</FONT><BR align=\"left\"/>", AfterColour);
    SmallString<80> UnchangedLineFormat = formatv(
        "<FONT COLOR=\"{0}\">%l</FONT><BR align=\"left\"/>", CommonColour);
    std::string Diff = Data[0]->getLabel().str();
    Diff += ":\n<BR align=\"left\"/>" +
            doSystemDiff(makeHTMLReady(SR[0]), makeHTMLReady(SR[1]),
                         OldLineFormat, NewLineFormat, UnchangedLineFormat);

    // Diff adds in some empty colour changes which are not valid HTML
    // so remove them.  Colours are all lowercase alpha characters (as
    // listed in https://graphviz.org/pdf/dotguide.pdf).
    Regex R("<FONT COLOR=\"\\w+\"></FONT>");
    while (true) {
      std::string Error;
      std::string S = R.sub("", Diff, &Error);
      if (Error != "")
        return Error;
      if (S == Diff)
        return Diff;
      Diff = S;
    }
    llvm_unreachable("Should not get here");
  }

  // Put node out in the appropriate colour.
  assert(!Data[1] && "Data[1] is set unexpectedly.");
  std::string Body = makeHTMLReady(Data[0]->getBody());
  const StringRef BS = Body;
  StringRef BS1 = BS;
  // Drop leading newline, if present.
  if (BS.front() == '\n')
    BS1 = BS1.drop_front(1);
  // Get label.
  StringRef Label = BS1.take_until([](char C) { return C == ':'; });
  // drop predecessors as they can be big and are redundant
  BS1 = BS1.drop_until([](char C) { return C == '\n'; }).drop_front();

  std::string S = "<FONT COLOR=\"" + Colour.str() + "\">" + Label.str() + ":";

  // align each line to the left.
  while (BS1.size()) {
    S.append("<BR align=\"left\"/>");
    StringRef Line = BS1.take_until([](char C) { return C == '\n'; });
    S.append(Line.str());
    BS1 = BS1.drop_front(Line.size() + 1);
  }
  S.append("<BR align=\"left\"/></FONT>");
  return S;
}

std::string DotCfgDiff::colourize(std::string S, StringRef Colour) const {
  if (S.length() == 0)
    return S;
  return "<FONT COLOR=\"" + Colour.str() + "\">" + S + "</FONT>";
}

DotCfgDiff::DotCfgDiff(StringRef Title, const FuncDataT<DCData> &Before,
                       const FuncDataT<DCData> &After)
    : GraphName(Title.str()) {
  StringMap<StringRef> EdgesMap;

  // Handle each basic block in the before IR.
  for (auto &B : Before.getData()) {
    StringRef Label = B.getKey();
    const BlockDataT<DCData> &BD = B.getValue();
    createNode(Label, BD, BeforeColour);

    // Create transitions with names made up of the from block label, the value
    // on which the transition is made and the to block label.
    for (StringMap<std::string>::const_iterator Sink = BD.getData().begin(),
                                                E = BD.getData().end();
         Sink != E; ++Sink) {
      std::string Key = (Label + " " + Sink->getKey().str()).str() + " " +
                        BD.getData().getSuccessorLabel(Sink->getKey()).str();
      EdgesMap.insert({Key, BeforeColour});
    }
  }

  // Handle each basic block in the after IR
  for (auto &A : After.getData()) {
    StringRef Label = A.getKey();
    const BlockDataT<DCData> &BD = A.getValue();
    unsigned C = NodePosition.count(Label);
    if (C == 0)
      // This only exists in the after IR.  Create the node.
      createNode(Label, BD, AfterColour);
    else {
      assert(C == 1 && "Unexpected multiple nodes.");
      Nodes[NodePosition[Label]].setCommon(BD);
    }
    // Add in the edges between the nodes (as common or only in after).
    for (StringMap<std::string>::const_iterator Sink = BD.getData().begin(),
                                                E = BD.getData().end();
         Sink != E; ++Sink) {
      std::string Key = (Label + " " + Sink->getKey().str()).str() + " " +
                        BD.getData().getSuccessorLabel(Sink->getKey()).str();
      unsigned C = EdgesMap.count(Key);
      if (C == 0)
        EdgesMap.insert({Key, AfterColour});
      else {
        EdgesMap[Key] = CommonColour;
      }
    }
  }

  // Now go through the map of edges and add them to the node.
  for (auto &E : EdgesMap) {
    // Extract the source, sink and value from the edge key.
    StringRef S = E.getKey();
    auto SP1 = S.rsplit(' ');
    auto &SourceSink = SP1.first;
    auto SP2 = SourceSink.split(' ');
    StringRef Source = SP2.first;
    StringRef Sink = SP2.second;
    StringRef Value = SP1.second;

    assert(NodePosition.count(Source) == 1 && "Expected to find node.");
    DotCfgDiffNode &SourceNode = Nodes[NodePosition[Source]];
    assert(NodePosition.count(Sink) == 1 && "Expected to find node.");
    unsigned SinkNode = NodePosition[Sink];
    StringRef Colour = E.second;

    // Look for an edge from Source to Sink
    if (EdgeLabels.count(SourceSink) == 0)
      EdgeLabels.insert({SourceSink, colourize(Value.str(), Colour)});
    else {
      StringRef V = EdgeLabels.find(SourceSink)->getValue();
      std::string NV = colourize(V.str() + " " + Value.str(), Colour);
      Colour = CommonColour;
      EdgeLabels[SourceSink] = NV;
    }
    SourceNode.addEdge(SinkNode, Value, Colour);
  }
  for (auto &I : Nodes)
    I.finalize(*this);
}

DotCfgDiffDisplayGraph DotCfgDiff::createDisplayGraph(StringRef Title,
                                                      StringRef EntryNodeName) {
  assert(NodePosition.count(EntryNodeName) == 1 &&
         "Expected to find entry block in map.");
  unsigned Entry = NodePosition[EntryNodeName];
  assert(Entry < Nodes.size() && "Expected to find entry node");
  DotCfgDiffDisplayGraph G(Title.str());

  std::map<const unsigned, unsigned> NodeMap;

  int EntryIndex = -1;
  unsigned Index = 0;
  for (auto &I : Nodes) {
    if (I.getIndex() == Entry)
      EntryIndex = Index;
    G.createNode(I.getBodyContent(), I.getColour());
    NodeMap.insert({I.getIndex(), Index++});
  }
  assert(EntryIndex >= 0 && "Expected entry node index to be set.");
  G.setEntryNode(EntryIndex);

  for (auto &I : NodeMap) {
    unsigned SourceNode = I.first;
    unsigned DisplayNode = I.second;
    getNode(SourceNode).createDisplayEdges(G, DisplayNode, NodeMap);
  }
  return G;
}

void DotCfgDiffNode::createDisplayEdges(
    DotCfgDiffDisplayGraph &DisplayGraph, unsigned DisplayNodeIndex,
    std::map<const unsigned, unsigned> &NodeMap) const {

  DisplayNode &SourceDisplayNode = DisplayGraph.getNode(DisplayNodeIndex);

  for (auto I : Edges) {
    unsigned SinkNodeIndex = I;
    StringRef Colour = getEdgeColour(SinkNodeIndex);
    const DotCfgDiffNode *SinkNode = &Graph.getNode(SinkNodeIndex);

    StringRef Label = Graph.getEdgeSourceLabel(getIndex(), SinkNodeIndex);
    DisplayNode &SinkDisplayNode = DisplayGraph.getNode(SinkNode->getIndex());
    SourceDisplayNode.createEdge(Label, SinkDisplayNode, Colour);
  }
  SourceDisplayNode.createEdgeMap();
}

void DotCfgDiffNode::finalize(DotCfgDiff &G) {
  for (auto E : EdgesMap) {
    Children.emplace_back(E.first);
    Edges.emplace_back(E.first);
  }
}

} // namespace

namespace llvm {

template <> struct GraphTraits<DotCfgDiffDisplayGraph *> {
  using NodeRef = const DisplayNode *;
  using ChildIteratorType = DisplayNode::ChildIterator;
  using nodes_iterator = DotCfgDiffDisplayGraph::NodeIterator;
  using EdgeRef = const DisplayEdge *;
  using ChildEdgeIterator = DisplayNode::EdgeIterator;

  static NodeRef getEntryNode(const DotCfgDiffDisplayGraph *G) {
    return G->getEntryNode();
  }
  static ChildIteratorType child_begin(NodeRef N) {
    return N->children_begin();
  }
  static ChildIteratorType child_end(NodeRef N) { return N->children_end(); }
  static nodes_iterator nodes_begin(const DotCfgDiffDisplayGraph *G) {
    return G->nodes_begin();
  }
  static nodes_iterator nodes_end(const DotCfgDiffDisplayGraph *G) {
    return G->nodes_end();
  }
  static ChildEdgeIterator child_edge_begin(NodeRef N) {
    return N->edges_begin();
  }
  static ChildEdgeIterator child_edge_end(NodeRef N) { return N->edges_end(); }
  static NodeRef edge_dest(EdgeRef E) { return &E->getDestinationNode(); }
  static unsigned size(const DotCfgDiffDisplayGraph *G) { return G->size(); }
};

template <>
struct DOTGraphTraits<DotCfgDiffDisplayGraph *> : public DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool Simple = false)
      : DefaultDOTGraphTraits(Simple) {}

  static bool renderNodesUsingHTML() { return true; }
  static std::string getGraphName(const DotCfgDiffDisplayGraph *DiffData) {
    return DiffData->getGraphName();
  }
  static std::string
  getGraphProperties(const DotCfgDiffDisplayGraph *DiffData) {
    return "\tsize=\"190, 190\";\n";
  }
  static std::string getNodeLabel(const DisplayNode *Node,
                                  const DotCfgDiffDisplayGraph *DiffData) {
    return DiffData->getNodeLabel(*Node);
  }
  static std::string getNodeAttributes(const DisplayNode *Node,
                                       const DotCfgDiffDisplayGraph *DiffData) {
    return DiffData->getNodeAttributes(*Node);
  }
  static std::string getEdgeSourceLabel(const DisplayNode *From,
                                        DisplayNode::ChildIterator &To) {
    return From->getEdgeSourceLabel(**To);
  }
  static std::string getEdgeAttributes(const DisplayNode *From,
                                       DisplayNode::ChildIterator &To,
                                       const DotCfgDiffDisplayGraph *DiffData) {
    return DiffData->getEdgeColorAttr(*From, **To);
  }
};

} // namespace llvm

namespace {

void DotCfgDiffDisplayGraph::generateDotFile(StringRef DotFile) {
  std::error_code EC;
  raw_fd_ostream OutStream(DotFile, EC);
  if (EC) {
    errs() << "Error: " << EC.message() << "\n";
    return;
  }
  WriteGraph(OutStream, this, false);
  OutStream.flush();
  OutStream.close();
}

} // namespace

namespace llvm {

DCData::DCData(const BasicBlock &B) {
  // Build up transition labels.
  const Instruction *Term = B.getTerminator();
  if (const BranchInst *Br = dyn_cast<const BranchInst>(Term))
    if (Br->isUnconditional())
      addSuccessorLabel(Br->getSuccessor(0)->getName().str(), "");
    else {
      addSuccessorLabel(Br->getSuccessor(0)->getName().str(), "true");
      addSuccessorLabel(Br->getSuccessor(1)->getName().str(), "false");
    }
  else if (const SwitchInst *Sw = dyn_cast<const SwitchInst>(Term)) {
    addSuccessorLabel(Sw->case_default()->getCaseSuccessor()->getName().str(),
                      "default");
    for (auto &C : Sw->cases()) {
      assert(C.getCaseValue() && "Expected to find case value.");
      SmallString<20> Value = formatv("{0}", C.getCaseValue()->getSExtValue());
      addSuccessorLabel(C.getCaseSuccessor()->getName().str(), Value);
    }
  } else
    for (const_succ_iterator I = succ_begin(&B), E = succ_end(&B); I != E; ++I)
      addSuccessorLabel((*I)->getName().str(), "");
}

DotCfgChangeReporter::DotCfgChangeReporter(bool Verbose)
    : ChangeReporter<IRDataT<DCData>>(Verbose) {}

void DotCfgChangeReporter::handleFunctionCompare(
    StringRef Name, StringRef Prefix, StringRef PassID, StringRef Divider,
    bool InModule, unsigned Minor, const FuncDataT<DCData> &Before,
    const FuncDataT<DCData> &After) {
  assert(HTML && "Expected outstream to be set");
  SmallString<8> Extender;
  SmallString<8> Number;
  // Handle numbering and file names.
  if (InModule) {
    Extender = formatv("{0}_{1}", N, Minor);
    Number = formatv("{0}.{1}", N, Minor);
  } else {
    Extender = formatv("{0}", N);
    Number = formatv("{0}", N);
  }
  // Create a temporary file name for the dot file.
  SmallVector<char, 128> SV;
  sys::fs::createUniquePath("cfgdot-%%%%%%.dot", SV, true);
  std::string DotFile = Twine(SV).str();

  SmallString<20> PDFFileName = formatv("diff_{0}.pdf", Extender);
  SmallString<200> Text;

  Text = formatv("{0}.{1}{2}{3}{4}", Number, Prefix, makeHTMLReady(PassID),
                 Divider, Name);

  DotCfgDiff Diff(Text, Before, After);
  std::string EntryBlockName = After.getEntryBlockName();
  // Use the before entry block if the after entry block was removed.
  if (EntryBlockName == "")
    EntryBlockName = Before.getEntryBlockName();
  assert(EntryBlockName != "" && "Expected to find entry block");

  DotCfgDiffDisplayGraph DG = Diff.createDisplayGraph(Text, EntryBlockName);
  DG.generateDotFile(DotFile);

  *HTML << genHTML(Text, DotFile, PDFFileName);
  std::error_code EC = sys::fs::remove(DotFile);
  if (EC)
    errs() << "Error: " << EC.message() << "\n";
}

std::string DotCfgChangeReporter::genHTML(StringRef Text, StringRef DotFile,
                                          StringRef PDFFileName) {
  SmallString<20> PDFFile = formatv("{0}/{1}", DotCfgDir, PDFFileName);
  // Create the PDF file.
  static ErrorOr<std::string> DotExe = sys::findProgramByName(DotBinary);
  if (!DotExe)
    return "Unable to find dot executable.";

  StringRef Args[] = {DotBinary, "-Tpdf", "-o", PDFFile, DotFile};
  int Result = sys::ExecuteAndWait(*DotExe, Args, None);
  if (Result < 0)
    return "Error executing system dot.";

  // Create the HTML tag refering to the PDF file.
  SmallString<200> S = formatv(
      "  <a href=\"{0}\" target=\"_blank\">{1}</a><br/>\n", PDFFileName, Text);
  return S.c_str();
}

void DotCfgChangeReporter::handleInitialIR(Any IR) {
  assert(HTML && "Expected outstream to be set");
  *HTML << "<button type=\"button\" class=\"collapsible\">0. "
        << "Initial IR (by function)</button>\n"
        << "<div class=\"content\">\n"
        << "  <p>\n";
  // Create representation of IR
  IRDataT<DCData> Data;
  IRComparer<DCData>::analyzeIR(IR, Data);
  // Now compare it against itself, which will have everything the
  // same and will generate the files.
  IRComparer<DCData>(Data, Data)
      .compare(getModuleForComparison(IR),
               [&](bool InModule, unsigned Minor,
                   const FuncDataT<DCData> &Before,
                   const FuncDataT<DCData> &After) -> void {
                 handleFunctionCompare("", " ", "Initial IR", "", InModule,
                                       Minor, Before, After);
               });
  *HTML << "  </p>\n"
        << "</div><br/>\n";
  ++N;
}

void DotCfgChangeReporter::generateIRRepresentation(Any IR, StringRef PassID,
                                                    IRDataT<DCData> &Data) {
  IRComparer<DCData>::analyzeIR(IR, Data);
}

void DotCfgChangeReporter::omitAfter(StringRef PassID, std::string &Name) {
  assert(HTML && "Expected outstream to be set");
  SmallString<20> Banner =
      formatv("  <a>{0}. Pass {1} on {2} omitted because no change</a><br/>\n",
              N, makeHTMLReady(PassID), Name);
  *HTML << Banner;
  ++N;
}

void DotCfgChangeReporter::handleAfter(StringRef PassID, std::string &Name,
                                       const IRDataT<DCData> &Before,
                                       const IRDataT<DCData> &After, Any IR) {
  assert(HTML && "Expected outstream to be set");
  IRComparer<DCData>(Before, After)
      .compare(getModuleForComparison(IR),
               [&](bool InModule, unsigned Minor,
                   const FuncDataT<DCData> &Before,
                   const FuncDataT<DCData> &After) -> void {
                 handleFunctionCompare(Name, " Pass ", PassID, " on ", InModule,
                                       Minor, Before, After);
               });
  *HTML << "    </p></div>\n";
  ++N;
}

void DotCfgChangeReporter::handleInvalidated(StringRef PassID) {
  assert(HTML && "Expected outstream to be set");
  SmallString<20> Banner =
      formatv("  <a>{0}. {1} invalidated</a><br/>\n", N, makeHTMLReady(PassID));
  *HTML << Banner;
  ++N;
}

void DotCfgChangeReporter::handleFiltered(StringRef PassID, std::string &Name) {
  assert(HTML && "Expected outstream to be set");
  SmallString<20> Banner =
      formatv("  <a>{0}. Pass {1} on {2} filtered out</a><br/>\n", N,
              makeHTMLReady(PassID), Name);
  *HTML << Banner;
  ++N;
}

void DotCfgChangeReporter::handleIgnored(StringRef PassID, std::string &Name) {
  assert(HTML && "Expected outstream to be set");
  SmallString<20> Banner = formatv("  <a>{0}. {1} on {2} ignored</a><br/>\n", N,
                                   makeHTMLReady(PassID), Name);
  *HTML << Banner;
  ++N;
}

bool DotCfgChangeReporter::initializeHTML() {
  std::error_code EC;
  HTML = std::make_unique<raw_fd_ostream>(DotCfgDir + "/passes.html", EC);
  if (EC) {
    HTML = nullptr;
    return false;
  }

  *HTML << "<!doctype html>"
        << "<html>"
        << "<head>"
        << "<style>.collapsible { "
        << "background-color: #777;"
        << " color: white;"
        << " cursor: pointer;"
        << " padding: 18px;"
        << " width: 100%;"
        << " border: none;"
        << " text-align: left;"
        << " outline: none;"
        << " font-size: 15px;"
        << "} .active, .collapsible:hover {"
        << " background-color: #555;"
        << "} .content {"
        << " padding: 0 18px;"
        << " display: none;"
        << " overflow: hidden;"
        << " background-color: #f1f1f1;"
        << "}"
        << "</style>"
        << "<title>passes.html</title>"
        << "</head>\n"
        << "<body>";
  return true;
}

DotCfgChangeReporter::~DotCfgChangeReporter() {
  if (!HTML)
    return;
  *HTML
      << "<script>var coll = document.getElementsByClassName(\"collapsible\");"
      << "var i;"
      << "for (i = 0; i < coll.length; i++) {"
      << "coll[i].addEventListener(\"click\", function() {"
      << " this.classList.toggle(\"active\");"
      << " var content = this.nextElementSibling;"
      << " if (content.style.display === \"block\"){"
      << " content.style.display = \"none\";"
      << " }"
      << " else {"
      << " content.style.display= \"block\";"
      << " }"
      << " });"
      << " }"
      << "</script>"
      << "</body>"
      << "</html>\n";
  HTML->flush();
  HTML->close();
}

void DotCfgChangeReporter::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  if ((PrintChanged == ChangePrinter::PrintChangedDotCfgVerbose ||
       PrintChanged == ChangePrinter::PrintChangedDotCfgQuiet)) {
    SmallString<128> OutputDir;
    sys::fs::expand_tilde(DotCfgDir, OutputDir);
    sys::fs::make_absolute(OutputDir);
    assert(!OutputDir.empty() && "expected output dir to be non-empty");
    DotCfgDir = OutputDir.c_str();
    if (initializeHTML()) {
      ChangeReporter<IRDataT<DCData>>::registerRequiredCallbacks(PIC);
      return;
    }
    dbgs() << "Unable to open output stream for -cfg-dot-changed\n";
  }
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
      WebsiteChangeReporter(PrintChanged ==
                            ChangePrinter::PrintChangedDotCfgVerbose),
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
  WebsiteChangeReporter.registerCallbacks(PIC);
}

template class ChangeReporter<std::string>;
template class TextChangeReporter<std::string>;

template class BlockDataT<EmptyData>;
template class FuncDataT<EmptyData>;
template class IRDataT<EmptyData>;
template class ChangeReporter<IRDataT<EmptyData>>;
template class TextChangeReporter<IRDataT<EmptyData>>;
template class IRComparer<EmptyData>;

} // namespace llvm
