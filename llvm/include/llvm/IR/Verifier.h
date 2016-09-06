//===- Verifier.h - LLVM IR Verifier ----------------------------*- C++ -*-===//
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

#ifndef LLVM_IR_VERIFIER_H
#define LLVM_IR_VERIFIER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class FunctionPass;
class ModulePass;
class Module;
class raw_ostream;

/// \brief Check a function for errors, useful for use when debugging a
/// pass.
///
/// If there are no errors, the function returns false. If an error is found,
/// a message describing the error is written to OS (if non-null) and true is
/// returned.
bool verifyFunction(const Function &F, raw_ostream *OS = nullptr);

/// \brief Check a module for errors.
///
/// If there are no errors, the function returns false. If an error is
/// found, a message describing the error is written to OS (if
/// non-null) and true is returned.
///
/// \return true if the module is broken. If BrokenDebugInfo is
/// supplied, DebugInfo verification failures won't be considered as
/// error and instead *BrokenDebugInfo will be set to true. Debug
/// info errors can be "recovered" from by stripping the debug info.
bool verifyModule(const Module &M, raw_ostream *OS = nullptr,
                  bool *BrokenDebugInfo = nullptr);

FunctionPass *createVerifierPass(bool FatalErrors = true);

/// Check a module for errors, and report separate error states for IR
/// and debug info errors.
class VerifierAnalysis : public AnalysisInfoMixin<VerifierAnalysis> {
  friend AnalysisInfoMixin<VerifierAnalysis>;
  static char PassID;

public:
  struct Result {
    bool IRBroken, DebugInfoBroken;
  };
  static void *ID() { return (void *)&PassID; }
  Result run(Module &M, ModuleAnalysisManager &);
  Result run(Function &F, FunctionAnalysisManager &);
};

/// Check a module for errors, but report debug info errors separately.
/// Otherwise behaves as the normal verifyModule. Debug info errors can be
/// "recovered" from by stripping the debug info.
bool verifyModule(bool &BrokenDebugInfo, const Module &M, raw_ostream *OS);

/// \brief Create a verifier pass.
///
/// Check a module or function for validity. This is essentially a pass wrapped
/// around the above verifyFunction and verifyModule routines and
/// functionality. When the pass detects a verification error it is always
/// printed to stderr, and by default they are fatal. You can override that by
/// passing \c false to \p FatalErrors.
///
/// Note that this creates a pass suitable for the legacy pass manager. It has
/// nothing to do with \c VerifierPass.
class VerifierPass : public PassInfoMixin<VerifierPass> {
  bool FatalErrors;

public:
  explicit VerifierPass(bool FatalErrors = true) : FatalErrors(FatalErrors) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};


} // End llvm namespace

#endif
