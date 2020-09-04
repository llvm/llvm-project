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

class Function;
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

// Base class for classes that report changes to the IR.
// It presents an interface for such classes and provides callbacks
// on various events as the new pass manager transforms the IR.
// It also provides filtering of information based on hidden options
// specifying which functions are interesting.
// Callbacks are made for the following events/queries:
// 1.  The initial IR processed.
// 2.  To get the representation of the IR (of type \p T).
// 3.  When a pass does not change the IR.
// 4.  When a pass changes the IR (given both before and after representations
//         of type \p T).
// 5.  When an IR is invalidated.
// 6.  When a pass is run on an IR that is not interesting (based on options).
// 7.  When a pass is ignored (pass manager or adapter pass).
// 8.  To compare two IR representations (of type \p T).
template <typename T> class ChangePrinter {
protected:
  ChangePrinter(
      std::function<void(Any IR)> HandleInitialIRFunc,
      std::function<void(Any IR, StringRef PassID, T &Output)>
          GenerateIRRepresentationFunc,
      std::function<void(StringRef PassID, std::string &Name)> OmitAfterFunc,
      std::function<void(StringRef PassID, std::string &Name, const T &Before,
                         const T &After, Any IR)>
          HandleAfterFunc,
      std::function<void(StringRef PassID)> HandleInvalidatedFunc,
      std::function<void(StringRef PassID, std::string &Name)>
          HandleFilteredFunc,
      std::function<void(StringRef PassID, std::string &Name)>
          HandleIgnoredFunc,
      std::function<bool(const T &Before, const T &After)> SameFunc)
      : HandleInitialIR(HandleInitialIRFunc),
        GenerateIRRepresentation(GenerateIRRepresentationFunc),
        OmitAfter(OmitAfterFunc), HandleAfter(HandleAfterFunc),
        HandleInvalidated(HandleInvalidatedFunc),
        HandleFiltered(HandleFilteredFunc), HandleIgnored(HandleIgnoredFunc),
        Same(SameFunc), InitialIR(true) {}

public:
  // Not virtual as classes are expected to be referenced as derived classes.
  ~ChangePrinter() {
    assert(BeforeStack.empty() && "Problem with Change Printer stack.");
  }

  // Determine if this pass/IR is interesting and if so, save the IR
  // otherwise it is left on the stack without data
  void saveIRBeforePass(Any IR, StringRef PassID);
  // Compare the IR from before the pass after the pass.
  void handleIRAfterPass(Any IR, StringRef PassID);
  // Handle the situation where a pass is invalidated.
  void handleInvalidatedPass(StringRef PassID);

private:
  // callback on the first IR processed
  std::function<void(Any IR)> HandleInitialIR;
  // callback before and after a pass to get the representation of the IR
  std::function<void(Any IR, StringRef PassID, T &Output)>
      GenerateIRRepresentation;
  // callback when the pass is not iteresting
  std::function<void(StringRef PassID, std::string &Name)> OmitAfter;
  // callback when interesting IR has changed
  std::function<void(StringRef PassID, std::string &Name, const T &Before,
                     const T &After, Any)>
      HandleAfter;
  // callback when an interesting pass is invalidated
  std::function<void(StringRef PassID)> HandleInvalidated;
  // callback when the IR or pass is not interesting
  std::function<void(StringRef PassID, std::string &Name)> HandleFiltered;
  // callback when an ignored pass is encountered
  std::function<void(StringRef PassID, std::string &Name)> HandleIgnored;
  // callback to compare the before and after representations of the IR
  std::function<bool(const T &Before, const T &After)> Same;

  // stack of IRs before passes
  std::vector<T> BeforeStack;
  // Is this the first IR seen?
  bool InitialIR;
};

// A change printer based on the string representation of the IR as created
// by unwrapAndPrint.  The string representation is stored in a std::string
// to preserve it as the IR changes in each pass.  Note that the banner is
// included in this representation but it is massaged before reporting.
class IRChangePrinter : public ChangePrinter<std::string> {
public:
  IRChangePrinter();
  void registerCallbacks(PassInstrumentationCallbacks &PIC);

protected:
  raw_ostream &Out;
};

/// This class provides an interface to register all the standard pass
/// instrumentations and manages their state (if any).
class StandardInstrumentations {
  PrintIRInstrumentation PrintIR;
  PrintPassInstrumentation PrintPass;
  TimePassesHandler TimePasses;
  OptNoneInstrumentation OptNone;
  IRChangePrinter PrintChangedIR;

public:
  StandardInstrumentations(bool DebugLogging) : PrintPass(DebugLogging) {}

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

  TimePassesHandler &getTimePasses() { return TimePasses; }
};
} // namespace llvm

#endif
