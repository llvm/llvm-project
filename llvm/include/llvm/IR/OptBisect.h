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

  /// Checks the bisect limit to determine if the specified pass should run.
  ///
  /// This function will immediate return true if bisection is disabled. If the
  /// bisect limit is set to -1, the function will print a message describing
  /// the pass and the bisect number assigned to it and return true.  Otherwise,
  /// the function will print a message with the bisect number assigned to the
  /// pass and indicating whether or not the pass will be run and return true if
  /// the bisect limit has not yet been exceded or false if it has.
  ///
  /// In order to avoid duplicating the code necessary to access OptBisect
  /// through the LLVMContext class, passes may call one of the helper
  /// functions that get the context from an IR object.  For instance,
  /// function passes may call skipPassForFunction().
  template <class UnitT>
  bool shouldRunPass(const StringRef PassName, const UnitT &U);

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

// Access to OptBisect should go through LLVMContext, but for the
// new pass manager there is no single base class from which a
// helper function to abstract the messy details can be provided.
// Instead, we provide standalone helper functions for each IR
// type that must be handled.

class Module;
class Function;
//class BasicBlock;
//class Loop;

/// Check with the OptBisect object to determine whether the described pass
/// should be skipped.
///
/// This is a helper function which abstracts the details of accessing OptBisect
/// through an LLVMContext obtained from a Module.
bool skipPassForModule(const StringRef PassName, const Module &M);

/// Check with the OptBisect object to determine whether the described pass
/// should be skipped.
///
/// This is a helper function which abstracts the details of accessing OptBisect
/// through an LLVMContext obtained from a Function.
bool skipPassForFunction(const StringRef PassName, const Function &F);
#if 0
/// Check with the OptBisect object to determine whether the described pass
/// should be skipped.
///
/// This is a helper function which abstracts the details of accessing OptBisect
/// through an LLVMContext obtained from a BasicBlock.
bool skipPassForBasicBlock(const StringRef PassName, const BasicBlock &BB);

/// Check with the OptBisect object to determine whether the described pass
/// should be skipped.
///
/// This is a helper function which abstracts the details of accessing OptBisect
/// through an LLVMContext obtained from a Loop.
bool skipPassForLoop(const StringRef PassName, const Loop &L);
#endif
// skiPassForSCC is declared in LazyCallGraph.h because of include file
// dependency issues related to LazyCallGraph::SCC being nested.

} // end namespace llvm

#endif // LLVM_IR_OPTBISECT_H
