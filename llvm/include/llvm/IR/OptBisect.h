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

#include "llvm/ADT/StringRef.h"
#include <limits>

namespace llvm {

class Pass;

/// Extensions to this class implement mechanisms to disable passes and
/// individual optimizations at compile time.
class OptPassGate {
public:
  virtual ~OptPassGate() = default;

  /// IRDescription is a textual description of the IR unit the pass is running
  /// over.
  virtual bool shouldRunPass(const Pass *P, StringRef IRDescription) {
    return true;
  }

  /// isEnabled() should return true before calling shouldRunPass().
  virtual bool isEnabled() const { return false; }
};

/// This class implements a mechanism to disable passes and individual
/// optimizations at compile time based on a command line option
/// (-opt-bisect-limit) in order to perform a bisecting search for
/// optimization-related problems.
class OptBisect : public OptPassGate {
public:
  /// Default constructor. Initializes the state to "disabled". The bisection
  /// will be enabled by the cl::opt call-back when the command line option
  /// is processed.
  /// Clients should not instantiate this class directly.  All access should go
  /// through LLVMContext.
  OptBisect() = default;

  virtual ~OptBisect() = default;

  /// Checks the bisect limit to determine if the specified pass should run.
  ///
  /// This forwards to checkPass().
  bool shouldRunPass(const Pass *P, StringRef IRDescription) override;

  /// isEnabled() should return true before calling shouldRunPass().
  bool isEnabled() const override { return BisectLimit != Disabled; }

  /// Set the new optimization limit and reset the counter. Passing
  /// OptBisect::Disabled disables the limiting.
  void setLimit(int Limit) {
    BisectLimit = Limit;
    LastBisectNum = 0;
  }

  /// Checks the bisect limit to determine if the specified pass should run.
  ///
  /// If the bisect limit is set to -1, the function prints a message describing
  /// the pass and the bisect number assigned to it and return true.  Otherwise,
  /// the function prints a message with the bisect number assigned to the
  /// pass and indicating whether or not the pass will be run and return true if
  /// the bisect limit has not yet been exceeded or false if it has.
  ///
  /// Most passes should not call this routine directly. Instead, they are
  /// called through helper routines provided by the pass base classes.  For
  /// instance, function passes should call FunctionPass::skipFunction().
  bool checkPass(const StringRef PassName, const StringRef TargetDesc);

  static const int Disabled = std::numeric_limits<int>::max();

private:
  int BisectLimit = Disabled;
  int LastBisectNum = 0;
};

/// Singleton instance of the OptBisect class, so multiple pass managers don't
/// need to coordinate their uses of OptBisect.
OptBisect &getOptBisector();

} // end namespace llvm

#endif // LLVM_IR_OPTBISECT_H
