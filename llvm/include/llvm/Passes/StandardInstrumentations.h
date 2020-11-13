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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/OptBisect.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/CommandLine.h"

#include <string>
#include <utility>

namespace llvm {

class Module;
class Function;

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

  using PrintModuleDesc = std::tuple<const Module *, std::string, StringRef>;

  void pushModuleDesc(StringRef PassID, Any IR);
  PrintModuleDesc popModuleDesc(StringRef PassID);

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

class OptBisectInstrumentation : public OptBisect {
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
private:
  // CFG is a map BB -> {(Succ, Multiplicity)}, where BB is a non-leaf basic
  // block, {(Succ, Multiplicity)} set of all pairs of the block's successors
  // and the multiplicity of the edge (BB->Succ). As the mapped sets are
  // unordered the order of successors is not tracked by the CFG. In other words
  // this allows basic block successors to be swapped by a pass without
  // reporting a CFG change. CFG can be guarded by basic block tracking pointers
  // in the Graph (BBGuard). That is if any of the block is deleted or RAUWed
  // then the CFG is treated poisoned and no block pointer of the Graph is used.
  struct CFG {
    struct BBGuard final : public CallbackVH {
      BBGuard(const BasicBlock *BB) : CallbackVH(BB) {}
      void deleted() override { CallbackVH::deleted(); }
      void allUsesReplacedWith(Value *) override { CallbackVH::deleted(); }
      bool isPoisoned() const { return !getValPtr(); }
    };

    Optional<DenseMap<intptr_t, BBGuard>> BBGuards;
    DenseMap<const BasicBlock *, DenseMap<const BasicBlock *, unsigned>> Graph;

    CFG(const Function *F, bool TrackBBLifetime = false);

    bool operator==(const CFG &G) const {
      return !isPoisoned() && !G.isPoisoned() && Graph == G.Graph;
    }

    bool isPoisoned() const {
      if (BBGuards)
        for (auto &BB : *BBGuards) {
          if (BB.second.isPoisoned())
            return true;
        }
      return false;
    }

    static void printDiff(raw_ostream &out, const CFG &Before,
                          const CFG &After);
  };

  SmallVector<std::pair<StringRef, Optional<CFG>>, 8> GraphStackBefore;

public:
  static cl::opt<bool> VerifyPreservedCFG;
  void registerCallbacks(PassInstrumentationCallbacks &PIC);
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
template <typename IRUnitT> class ChangePrinter {
protected:
  ChangePrinter() {}

public:
  virtual ~ChangePrinter();

  // Determine if this pass/IR is interesting and if so, save the IR
  // otherwise it is left on the stack without data.
  void saveIRBeforePass(Any IR, StringRef PassID);
  // Compare the IR from before the pass after the pass.
  void handleIRAfterPass(Any IR, StringRef PassID);
  // Handle the situation where a pass is invalidated.
  void handleInvalidatedPass(StringRef PassID);

protected:
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
};

// A change printer based on the string representation of the IR as created
// by unwrapAndPrint.  The string representation is stored in a std::string
// to preserve it as the IR changes in each pass.  Note that the banner is
// included in this representation but it is massaged before reporting.
class IRChangePrinter : public ChangePrinter<std::string> {
public:
  IRChangePrinter();
  ~IRChangePrinter() override;
  void registerCallbacks(PassInstrumentationCallbacks &PIC);

protected:
  // Called on the first IR processed.
  void handleInitialIR(Any IR) override;
  // Called before and after a pass to get the representation of the IR.
  void generateIRRepresentation(Any IR, StringRef PassID,
                                std::string &Output) override;
  // Called when the pass is not iteresting.
  void omitAfter(StringRef PassID, std::string &Name) override;
  // Called when an interesting IR has changed.
  void handleAfter(StringRef PassID, std::string &Name,
                   const std::string &Before, const std::string &After,
                   Any) override;
  // Called when an interesting pass is invalidated.
  void handleInvalidated(StringRef PassID) override;
  // Called when the IR or pass is not interesting.
  void handleFiltered(StringRef PassID, std::string &Name) override;
  // Called when an ignored pass is encountered.
  void handleIgnored(StringRef PassID, std::string &Name) override;
  // Called to compare the before and after representations of the IR.
  bool same(const std::string &Before, const std::string &After) override;

  raw_ostream &Out;
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
  IRChangePrinter PrintChangedIR;
  VerifyInstrumentation Verify;

  bool VerifyEach;

public:
  StandardInstrumentations(bool DebugLogging, bool VerifyEach = false)
      : PrintPass(DebugLogging), OptNone(DebugLogging), Verify(DebugLogging),
        VerifyEach(VerifyEach) {}

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

  TimePassesHandler &getTimePasses() { return TimePasses; }
};
} // namespace llvm

#endif
