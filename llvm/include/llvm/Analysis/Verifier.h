//===-- llvm/Analysis/Verifier.h - LLVM IR Verifier -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the function verifier interface, that can be used for some
// sanity checking of input to the system, and for checking that transformations
// haven't done something bad.
//
// Note that this does not provide full 'java style' security and verifications,
// instead it just tries to ensure that code is well formed.
//
// To see what specifically is checked, look at the top of Verifier.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VERIFIER_H
#define LLVM_ANALYSIS_VERIFIER_H

#include <string>

namespace llvm {

class FunctionPass;
class Module;
class Function;

/// @brief An enumeration to specify the action to be taken if errors found.
///
/// This enumeration is used in the functions below to indicate what should
/// happen if the verifier finds errors. Each of the functions that uses
/// this enumeration as an argument provides a default value for it. The
/// actions are listed below.
enum VerifierFailureAction {
  AbortProcessAction,   ///< verifyModule will print to stderr and abort()
  PrintMessageAction,   ///< verifyModule will print to stderr and return true
  ReturnStatusAction    ///< verifyModule will just return true
};

/// @brief Create a verifier pass.
///
/// Check a module or function for validity.  When the pass is used, the
/// action indicated by the \p action argument will be used if errors are
/// found.
FunctionPass *createVerifierPass(
  VerifierFailureAction action = AbortProcessAction ///< Action to take
);

/// @brief Check a module for errors.
///
/// If there are no errors, the function returns false. If an error is found,
/// the action taken depends on the \p action parameter.
/// This should only be used for debugging, because it plays games with
/// PassManagers and stuff.

bool verifyModule(
  const Module &M,  ///< The module to be verified
  VerifierFailureAction action = AbortProcessAction, ///< Action to take
  std::string *ErrorInfo = 0      ///< Information about failures.
);

// verifyFunction - Check a function for errors, useful for use when debugging a
// pass.
bool verifyFunction(
  const Function &F,  ///< The function to be verified
  VerifierFailureAction action = AbortProcessAction ///< Action to take
);

} // End llvm namespace

#endif
