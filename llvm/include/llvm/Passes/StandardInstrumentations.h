//===- StandardInstrumentations.h ------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header defines a class that provides bookkeeping for all standard
/// (i.e in-tree) pass instrumentations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_STANDARDINSTRUMENTATIONS_H
#define LLVM_PASSES_STANDARDINSTRUMENTATIONS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/OptBisect.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO/SampleProfileProbe.h"

#include <string>
#include <utility>

namespace llvm {

class Module;
class Function;
class PassInstrumentationCallbacks;

/// Instrumentation to print IR before/after passes.
///
/// Needs state to be able to print module after pass that invalidates IR unit
/// (typically Loop or SCC).
class PrintIRInstrumentation {
public:
  ~PrintIRInstrumentation();

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

private:
  void printBeforePass(StringRef PassID, Any IR);
  void printAfterPass(StringRef PassID, Any IR);
  void printAfterPassInvalidated(StringRef PassID);

  bool shouldPrintBeforePass(StringRef PassID);
  bool shouldPrintAfterPass(StringRef PassID);

  using PrintModuleDesc = std::tuple<const Module *, std::string, StringRef>;

  void pushModuleDesc(StringRef PassID, Any IR);
  PrintModuleDesc popModuleDesc(StringRef PassID);

  PassInstrumentationCallbacks *PIC;
  /// Stack of Module description, enough to print the module after a given
  /// pass.
  SmallVector<PrintModuleDesc, 2> ModuleDescStack;
  bool StoreModuleDesc = false;
};

class OptNoneInstrumentation {
public:
  OptNoneInstrumentation(bool DebugLogging) : DebugLogging(DebugLogging) {}
  void registerCallbacks(PassInstrumentationCallbacks &PIC);

private:
  bool DebugLogging;
  bool shouldRun(StringRef PassID, Any IR);
};

class OptBisectInstrumentation {
public:
  OptBisectInstrumentation() {}
  void registerCallbacks(PassInstrumentationCallbacks &PIC);
};

// Debug logging for transformation and analysis passes.
class PrintPassInstrumentation {
public:
  PrintPassInstrumentation(bool DebugLogging) : DebugLogging(DebugLogging) {}
  void registerCallbacks(PassInstrumentationCallbacks &PIC);

private:
  bool DebugLogging;
};

class PreservedCFGCheckerInstrumentation {
public:
  // Keeps sticky poisoned flag for the given basic block once it has been
  // deleted or RAUWed.
  struct BBGuard final : public CallbackVH {
    BBGuard(const BasicBlock *BB) : CallbackVH(BB) {}
    void deleted() override { CallbackVH::deleted(); }
    void allUsesReplacedWith(Value *) override { CallbackVH::deleted(); }
    bool isPoisoned() const { return !getValPtr(); }
  };

  // CFG is a map BB -> {(Succ, Multiplicity)}, where BB is a non-leaf basic
  // block, {(Succ, Multiplicity)} set of all pairs of the block's successors
  // and the multiplicity of the edge (BB->Succ). As the mapped sets are
  // unordered the order of successors is not tracked by the CFG. In other words
  // this allows basic block successors to be swapped by a pass without
  // reporting a CFG change. CFG can be guarded by basic block tracking pointers
  // in the Graph (BBGuard). That is if any of the block is deleted or RAUWed
  // then the CFG is treated poisoned and no block pointer of the Graph is used.
  struct CFG {
    Optional<DenseMap<intptr_t, BBGuard>> BBGuards;
    DenseMap<const BasicBlock *, DenseMap<const BasicBlock *, unsigned>> Graph;

    CFG(const Function *F, bool TrackBBLifetime);

    bool operator==(const CFG &G) const {
      return !isPoisoned() && !G.isPoisoned() && Graph == G.Graph;
    }

    bool isPoisoned() const {
      return BBGuards &&
             std::any_of(BBGuards->begin(), BBGuards->end(),
                         [](const auto &BB) { return BB.second.isPoisoned(); });
    }

    static void printDiff(raw_ostream &out, const CFG &Before,
                          const CFG &After);
    bool invalidate(Function &F, const PreservedAnalyses &PA,
                    FunctionAnalysisManager::Invalidator &);
  };

#ifndef NDEBUG
  SmallVector<StringRef, 8> PassStack;
#endif

  static cl::opt<bool> VerifyPreservedCFG;
  void registerCallbacks(PassInstrumentationCallbacks &PIC,
                         FunctionAnalysisManager &FAM);
};

// Base class for classes that report changes to the IR.
// It presents an interface for such classes and provides calls
// on various events as the new pass manager transforms the IR.
// It also provides filtering of information based on hidden options
// specifying which functions are interesting.
// Calls are made for the following events/queries:
// 1.  The initial IR processed.
// 2.  To get the representation of the IR (of type \p T).
// 3.  When a pass does not change the IR.
// 4.  When a pass changes the IR (given both before and after representations
//         of type \p T).
// 5.  When an IR is invalidated.
// 6.  When a pass is run on an IR that is not interesting (based on options).
// 7.  When a pass is ignored (pass manager or adapter pass).
// 8.  To compare two IR representations (of type \p T).
template <typename IRUnitT> class ChangeReporter {
protected:
  ChangeReporter(bool RunInVerboseMode) : VerboseMode(RunInVerboseMode) {}

public:
  virtual ~ChangeReporter();

  // Determine if this pass/IR is interesting and if so, save the IR
  // otherwise it is left on the stack without data.
  void saveIRBeforePass(Any IR, StringRef PassID);
  // Compare the IR from before the pass after the pass.
  void handleIRAfterPass(Any IR, StringRef PassID);
  // Handle the situation where a pass is invalidated.
  void handleInvalidatedPass(StringRef PassID);

protected:
  // Register required callbacks.
  void registerRequiredCallbacks(PassInstrumentationCallbacks &PIC);

  // Return true when this is a defined function for which printing
  // of changes is desired.
  bool isInterestingFunction(const Function &F);

  // Return true when this is a pass for which printing of changes is desired.
  bool isInterestingPass(StringRef PassID);

  // Return true when this is a pass on IR for which printing
  // of changes is desired.
  bool isInteresting(Any IR, StringRef PassID);

  // Called on the first IR processed.
  virtual void handleInitialIR(Any IR) = 0;
  // Called before and after a pass to get the representation of the IR.
  virtual void generateIRRepresentation(Any IR, StringRef PassID,
                                        IRUnitT &Output) = 0;
  // Called when the pass is not iteresting.
  virtual void omitAfter(StringRef PassID, std::string &Name) = 0;
  // Called when an interesting IR has changed.
  virtual void handleAfter(StringRef PassID, std::string &Name,
                           const IRUnitT &Before, const IRUnitT &After,
                           Any) = 0;
  // Called when an interesting pass is invalidated.
  virtual void handleInvalidated(StringRef PassID) = 0;
  // Called when the IR or pass is not interesting.
  virtual void handleFiltered(StringRef PassID, std::string &Name) = 0;
  // Called when an ignored pass is encountered.
  virtual void handleIgnored(StringRef PassID, std::string &Name) = 0;
  // Called to compare the before and after representations of the IR.
  virtual bool same(const IRUnitT &Before, const IRUnitT &After) = 0;

  // Stack of IRs before passes.
  std::vector<IRUnitT> BeforeStack;
  // Is this the first IR seen?
  bool InitialIR = true;

  // Run in verbose mode, printing everything?
  const bool VerboseMode;
};

// An abstract template base class that handles printing banners and
// reporting when things have not changed or are filtered out.
template <typename IRUnitT>
class TextChangeReporter : public ChangeReporter<IRUnitT> {
protected:
  TextChangeReporter(bool Verbose);

  // Print a module dump of the first IR that is changed.
  void handleInitialIR(Any IR) override;
  // Report that the IR was omitted because it did not change.
  void omitAfter(StringRef PassID, std::string &Name) override;
  // Report that the pass was invalidated.
  void handleInvalidated(StringRef PassID) override;
  // Report that the IR was filtered out.
  void handleFiltered(StringRef PassID, std::string &Name) override;
  // Report that the pass was ignored.
  void handleIgnored(StringRef PassID, std::string &Name) override;
  // Make substitutions in \p S suitable for reporting changes
  // after the pass and then print it.

  raw_ostream &Out;
};

// A change printer based on the string representation of the IR as created
// by unwrapAndPrint.  The string representation is stored in a std::string
// to preserve it as the IR changes in each pass.  Note that the banner is
// included in this representation but it is massaged before reporting.
class IRChangedPrinter : public TextChangeReporter<std::string> {
public:
  IRChangedPrinter(bool VerboseMode)
      : TextChangeReporter<std::string>(VerboseMode) {}
  ~IRChangedPrinter() override;
  void registerCallbacks(PassInstrumentationCallbacks &PIC);

protected:
  // Called before and after a pass to get the representation of the IR.
  void generateIRRepresentation(Any IR, StringRef PassID,
                                std::string &Output) override;
  // Called when an interesting IR has changed.
  void handleAfter(StringRef PassID, std::string &Name,
                   const std::string &Before, const std::string &After,
                   Any) override;
  // Called to compare the before and after representations of the IR.
  bool same(const std::string &Before, const std::string &After) override;
};

// The following classes hold a representation of the IR for a change
// reporter that uses string comparisons of the basic blocks
// that are created using print (ie, similar to dump()).
// These classes respect the filtering of passes and functions using
// -filter-passes and -filter-print-funcs.
//
// Information that needs to be saved for a basic block in order to compare
// before and after the pass to determine if it was changed by a pass.
class ChangedBlockData {
public:
  ChangedBlockData(const BasicBlock &B);

  bool operator==(const ChangedBlockData &That) const {
    return Body == That.Body;
  }
  bool operator!=(const ChangedBlockData &That) const {
    return Body != That.Body;
  }

  // Return the label of the represented basic block.
  StringRef getLabel() const { return Label; }
  // Return the string representation of the basic block.
  StringRef getBody() const { return Body; }

protected:
  std::string Label;
  std::string Body;
};

template <typename IRData> class OrderedChangedData {
public:
  // Return the names in the order they were saved
  std::vector<std::string> &getOrder() { return Order; }
  const std::vector<std::string> &getOrder() const { return Order; }

  // Return a map of names to saved representations
  StringMap<IRData> &getData() { return Data; }
  const StringMap<IRData> &getData() const { return Data; }

  bool operator==(const OrderedChangedData<IRData> &That) const {
    return Data == That.getData();
  }

  // Call the lambda \p HandlePair on each corresponding pair of data from
  // \p Before and \p After.  The order is based on the order in \p After
  // with ones that are only in \p Before interspersed based on where they
  // occur in \p Before.  This is used to present the output in an order
  // based on how the data is ordered in LLVM.
  static void
  report(const OrderedChangedData &Before, const OrderedChangedData &After,
         function_ref<void(const IRData *, const IRData *)> HandlePair);

protected:
  std::vector<std::string> Order;
  StringMap<IRData> Data;
};

// The data saved for comparing functions.
using ChangedFuncData = OrderedChangedData<ChangedBlockData>;

// A map of names to the saved data.
using ChangedIRData = OrderedChangedData<ChangedFuncData>;

// A class that compares two IRs and does a diff between them.  The
// added lines are prefixed with a '+', the removed lines are prefixed
// with a '-' and unchanged lines are prefixed with a space (to have
// things line up).
class ChangedIRComparer {
public:
  ChangedIRComparer(raw_ostream &OS, const ChangedIRData &Before,
                    const ChangedIRData &After, bool ColourMode)
      : Before(Before), After(After), Out(OS), UseColour(ColourMode) {}

  // Compare the 2 IRs.
  void compare(Any IR, StringRef Prefix, StringRef PassID, StringRef Name);

  // Analyze \p IR and build the IR representation in \p Data.
  static void analyzeIR(Any IR, ChangedIRData &Data);

protected:
  // Return the module when that is the appropriate level of
  // comparison for \p IR.
  static const Module *getModuleForComparison(Any IR);

  // Generate the data for \p F into \p Data.
  static bool generateFunctionData(ChangedIRData &Data, const Function &F);

  // Called to handle the compare of a function. When \p InModule is set,
  // this function is being handled as part of comparing a module.
  void handleFunctionCompare(StringRef Name, StringRef Prefix, StringRef PassID,
                             bool InModule, const ChangedFuncData &Before,
                             const ChangedFuncData &After);

  const ChangedIRData &Before;
  const ChangedIRData &After;
  raw_ostream &Out;
  bool UseColour;
};

// A change printer that prints out in-line differences in the basic
// blocks.  It uses an InlineComparer to do the comparison so it shows
// the differences prefixed with '-' and '+' for code that is removed
// and added, respectively.  Changes to the IR that do not affect basic
// blocks are not reported as having changed the IR.  The option
// -print-module-scope does not affect this change reporter.
class InLineChangePrinter : public TextChangeReporter<ChangedIRData> {
public:
  InLineChangePrinter(bool VerboseMode, bool ColourMode)
      : TextChangeReporter<ChangedIRData>(VerboseMode), UseColour(ColourMode) {}
  ~InLineChangePrinter() override;
  void registerCallbacks(PassInstrumentationCallbacks &PIC);

protected:
  // Create a representation of the IR.
  virtual void generateIRRepresentation(Any IR, StringRef PassID,
                                        ChangedIRData &Output) override;

  // Called when an interesting IR has changed.
  virtual void handleAfter(StringRef PassID, std::string &Name,
                           const ChangedIRData &Before,
                           const ChangedIRData &After, Any) override;
  // Called to compare the before and after representations of the IR.
  virtual bool same(const ChangedIRData &Before,
                    const ChangedIRData &After) override;

  bool UseColour;
};

class VerifyInstrumentation {
  bool DebugLogging;

public:
  VerifyInstrumentation(bool DebugLogging) : DebugLogging(DebugLogging) {}
  void registerCallbacks(PassInstrumentationCallbacks &PIC);
};

/// This class provides an interface to register all the standard pass
/// instrumentations and manages their state (if any).
class StandardInstrumentations {
  PrintIRInstrumentation PrintIR;
  PrintPassInstrumentation PrintPass;
  TimePassesHandler TimePasses;
  OptNoneInstrumentation OptNone;
  OptBisectInstrumentation OptBisect;
  PreservedCFGCheckerInstrumentation PreservedCFGChecker;
  IRChangedPrinter PrintChangedIR;
  PseudoProbeVerifier PseudoProbeVerification;
  InLineChangePrinter PrintChangedDiff;
  VerifyInstrumentation Verify;

  bool VerifyEach;

public:
  StandardInstrumentations(bool DebugLogging, bool VerifyEach = false);

  // Register all the standard instrumentation callbacks. If \p FAM is nullptr
  // then PreservedCFGChecker is not enabled.
  void registerCallbacks(PassInstrumentationCallbacks &PIC,
                         FunctionAnalysisManager *FAM = nullptr);

  TimePassesHandler &getTimePasses() { return TimePasses; }
};

extern template class ChangeReporter<std::string>;
extern template class TextChangeReporter<std::string>;

extern template class ChangeReporter<ChangedIRData>;
extern template class TextChangeReporter<ChangedIRData>;

} // namespace llvm

#endif
