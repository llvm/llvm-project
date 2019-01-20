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

namespace llvm {

class Pass;
class Module;
class Function;
class BasicBlock;
class Region;
class Loop;
class CallGraphSCC;

/// Extensions to this class implement mechanisms to disable passes and
/// individual optimizations at compile time.
class OptPassGate {
public:
  virtual ~OptPassGate() = default;

  virtual bool shouldRunPass(const Pass *P, const Module &U) { return true; }
  virtual bool shouldRunPass(const Pass *P, const Function &U)  {return true; }
  virtual bool shouldRunPass(const Pass *P, const BasicBlock &U)  { return true; }
  virtual bool shouldRunPass(const Pass *P, const Region &U)  { return true; }
  virtual bool shouldRunPass(const Pass *P, const Loop &U)  { return true; }
  virtual bool shouldRunPass(const Pass *P, const CallGraphSCC &U)  { return true; }
};

/// This class implements a mechanism to disable passes and individual
/// optimizations at compile time based on a command line option
/// (-opt-bisect-limit) in order to perform a bisecting search for
/// optimization-related problems.
class OptBisect : public OptPassGate {
public:
  /// Default constructor, initializes the OptBisect state based on the
  /// -opt-bisect-limit command line argument.
  ///
  /// By default, bisection is disabled.
  ///
  /// Clients should not instantiate this class directly.  All access should go
  /// through LLVMContext.
  OptBisect();

  virtual ~OptBisect() = default;

  /// Checks the bisect limit to determine if the specified pass should run.
  ///
  /// These functions immediately return true if bisection is disabled. If the
  /// bisect limit is set to -1, the functions print a message describing
  /// the pass and the bisect number assigned to it and return true.  Otherwise,
  /// the functions print a message with the bisect number assigned to the
  /// pass and indicating whether or not the pass will be run and return true if
  /// the bisect limit has not yet been exceeded or false if it has.
  ///
  /// Most passes should not call these routines directly. Instead, they are
  /// called through helper routines provided by the pass base classes.  For
  /// instance, function passes should call FunctionPass::skipFunction().
  bool shouldRunPass(const Pass *P, const Module &U) override;
  bool shouldRunPass(const Pass *P, const Function &U) override;
  bool shouldRunPass(const Pass *P, const BasicBlock &U) override;
  bool shouldRunPass(const Pass *P, const Region &U) override;
  bool shouldRunPass(const Pass *P, const Loop &U) override;
  bool shouldRunPass(const Pass *P, const CallGraphSCC &U) override;

private:
  bool checkPass(const StringRef PassName, const StringRef TargetDesc);

  bool BisectEnabled = false;
  unsigned LastBisectNum = 0;
};

} // end namespace llvm

#endif // LLVM_IR_OPTBISECT_H
