//===----------- llvm/IR/OptBisect.h - LLVM Bisect support -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the interface for bisecting optimizations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_OPTBISECT_H
#define LLVM_IR_OPTBISECT_H

namespace llvm {

class Pass;
class StringRef;
class Twine;

/// This class implements a mechanism to disable passes and individual
/// optimizations at compile time based on a command line option
/// (-opt-bisect-limit) in order to perform a bisecting search for
/// optimization-related problems.
class OptBisect {
public:
  /// \brief Default constructor, initializes the OptBisect state based on the
  /// -opt-bisect-limit command line argument.
  ///
  /// By default, bisection is disabled.
  ///
  /// Clients should not instantiate this class directly.  All access should go
  /// through LLVMContext.
  OptBisect();

  /// Checks the bisect limit to determine if the specified pass should run.
  ///
  /// This function will immediate return true if bisection is disabled. If the
  /// bisect limit is set to -1, the function will print a message describing
  /// the pass and the bisect number assigned to it and return true.  Otherwise,
  /// the function will print a message with the bisect number assigned to the
  /// pass and indicating whether or not the pass will be run and return true if
  /// the bisect limit has not yet been exceded or false if it has.
  ///
  /// Most passes should not call this routine directly.  Instead, it is called
  /// through a helper routine provided by the pass base class.  For instance,
  /// function passes should call FunctionPass::skipFunction().
  template <class UnitT>
  bool shouldRunPass(const Pass *P, const UnitT &U);

  /// Checks the bisect limit to determine if the optimization described by the
  /// /p Desc argument should run.
  ///
  /// This function will immediate return true if bisection is disabled. If the
  /// bisect limit is set to -1, the function will print a message with the
  /// bisect number assigned to the optimization along with the /p Desc
  /// description and return true.  Otherwise, the function will print a message
  /// with the bisect number assigned to the optimization and indicating whether
  /// or not the pass will be run and return true if the bisect limit has not
  /// yet been exceded or false if it has.
  ///
  /// Passes may call this function to provide more fine grained control over
  /// individual optimizations performed by the pass.  Passes which cannot be
  /// skipped entirely (such as non-optional code generation passes) may still
  /// call this function to control whether or not individual optional
  /// transformations are performed.
  bool shouldRunCase(const Twine &Desc);

private:
  bool checkPass(const StringRef PassName, const StringRef TargetDesc);

  bool BisectEnabled = false;
  unsigned LastBisectNum = 0;
};

} // end namespace llvm

#endif // LLVM_IR_OPTBISECT_H
