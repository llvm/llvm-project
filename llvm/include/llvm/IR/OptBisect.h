//===- llvm/IR/OptBisect.h - LLVM Bisect support ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the interface for bisecting optimizations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_OPTBISECT_H
#define LLVM_IR_OPTBISECT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Range.h"

namespace llvm {

/// Extensions to this class implement mechanisms to disable passes and
/// individual optimizations at compile time.
class OptPassGate {
public:
  virtual ~OptPassGate() = default;

  /// IRDescription is a textual description of the IR unit the pass is running
  /// over.
  virtual bool shouldRunPass(StringRef PassName,
                             StringRef IRDescription) const {
    return true;
  }

  /// isEnabled() should return true before calling shouldRunPass().
  virtual bool isEnabled() const { return false; }
};

/// This class implements a mechanism to disable passes and individual
/// optimizations at compile time based on a command line option
/// (-opt-bisect) in order to perform a bisecting search for
/// optimization-related problems.
class LLVM_ABI OptBisect : public OptPassGate {
public:
  /// Default constructor. Initializes the state to "disabled". The bisection
  /// will be enabled by the cl::opt call-back when the command line option
  /// is processed.
  /// Clients should not instantiate this class directly.  All access should go
  /// through LLVMContext.
  OptBisect() = default;

  ~OptBisect() override = default;

  /// Checks the bisect ranges to determine if the specified pass should run.
  ///
  /// The method prints the name of the pass, its assigned bisect number, and
  /// whether or not the pass will be executed. It returns true if the pass
  /// should run, i.e. if no ranges are specified or the current pass number
  /// falls within one of the specified ranges.
  ///
  /// Most passes should not call this routine directly. Instead, it is called
  /// through helper routines provided by the base classes of the pass. For
  /// instance, function passes should call FunctionPass::skipFunction().
  bool shouldRunPass(StringRef PassName,
                     StringRef IRDescription) const override;

  /// isEnabled() should return true before calling shouldRunPass().
  bool isEnabled() const override { return !BisectRanges.empty(); }

  /// Set ranges directly from a RangeList.
  void setRanges(RangeUtils::RangeList Ranges) {
    BisectRanges = std::move(Ranges);
  }

  /// Clear all ranges, effectively disabling bisection.
  void clearRanges() {
    BisectRanges.clear();
    LastBisectNum = 0;
  }

private:
  mutable int LastBisectNum = 0;
  RangeUtils::RangeList BisectRanges;
};

/// This class implements a mechanism to disable passes and individual
/// optimizations at compile time based on a command line option
/// (-opt-disable) in order to study how single transformations, or
/// combinations thereof, affect the IR.
class LLVM_ABI OptDisable : public OptPassGate {
public:
  /// Checks the pass name to determine if the specified pass should run.
  ///
  /// It returns true if the pass should run, i.e. if its name is was
  /// not provided via command line.
  /// If -opt-disable-enable-verbosity is given, the method prints the
  /// name of the pass, and whether or not the pass will be executed.
  ///
  /// Most passes should not call this routine directly. Instead, it is called
  /// through helper routines provided by the base classes of the pass. For
  /// instance, function passes should call FunctionPass::skipFunction().
  bool shouldRunPass(StringRef PassName,
                     StringRef IRDescription) const override;

  /// Parses the command line argument to extract the names of the passes
  /// to be disabled. Multiple pass names can be provided with comma separation.
  void setDisabled(StringRef Pass);

  /// isEnabled() should return true before calling shouldRunPass().
  bool isEnabled() const override { return !DisabledPasses.empty(); }

private:
  StringSet<> DisabledPasses = {};
};

/// Singleton instance of the OptPassGate class, so multiple pass managers don't
/// need to coordinate their uses of OptBisect and OptDisable.
LLVM_ABI OptPassGate &getGlobalPassGate();

} // end namespace llvm

#endif // LLVM_IR_OPTBISECT_H
