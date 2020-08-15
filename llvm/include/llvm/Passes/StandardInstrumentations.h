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
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassTimingInfo.h"

#include <string>
#include <utility>

namespace llvm {

class Module;

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

/// This class provides an interface to register all the standard pass
/// instrumentations and manages their state (if any).
class StandardInstrumentations {
  PrintIRInstrumentation PrintIR;
  PrintPassInstrumentation PrintPass;
  TimePassesHandler TimePasses;
  OptNoneInstrumentation OptNone;

public:
  StandardInstrumentations(bool DebugLogging) : PrintPass(DebugLogging) {}

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

  TimePassesHandler &getTimePasses() { return TimePasses; }
};
} // namespace llvm

#endif
