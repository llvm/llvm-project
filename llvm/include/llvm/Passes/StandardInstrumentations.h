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
  PrintIRInstrumentation() = default;
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
  OptNoneInstrumentation() {}
  void registerCallbacks(PassInstrumentationCallbacks &PIC);

private:
  bool skip(StringRef PassID, Any IR);
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

/// This class provides an interface to register all the standard pass
/// instrumentations and manages their state (if any).
class StandardInstrumentations {
  PrintIRInstrumentation PrintIR;
  PrintPassInstrumentation PrintPass;
  TimePassesHandler TimePasses;
  OptNoneInstrumentation OptNone;
  PreservedCFGCheckerInstrumentation PreservedCFGChecker;

public:
  StandardInstrumentations(bool DebugLogging) : PrintPass(DebugLogging) {}

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

  TimePassesHandler &getTimePasses() { return TimePasses; }
};
} // namespace llvm

#endif
