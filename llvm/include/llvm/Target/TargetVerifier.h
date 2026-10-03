//===-- llvm/Target/TargetVerifier.h - Target IR Verifier -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a target-dependent verification framework that can be
// instantiated and run from target-independent locations.
//
// In contrast to the generic Verifier or Lint, a TargetVerify looks for
// constructs that are invalid for a *particular* target machine. Each backend
// provides its own TargetVerify and registers a factory for it (keyed by
// Triple::ArchType) via registerTargetVerify(), typically from its
// LLVMInitialize<Target>Target().
//
// TargetVerifierPass is the target-independent dispatcher: it reads the
// module's target triple and runs whichever TargetVerify (if any) was
// registered for that architecture. Because dispatch happens at run time by
// triple, the pass can be scheduled from generic, target-independent pipelines
// (e.g. `opt -passes=target-verifier`); it is simply a no-op for targets that
// have not registered a verifier.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETVERIFIER_H
#define LLVM_TARGET_TARGETVERIFIER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <string>

namespace llvm {

class Function;
class ModulePass;
class PassRegistry;

/// Base class for a single target's IR verification logic.
///
/// A backend subclasses this and implements run() to check a function for
/// constructs that are invalid for that target. Failed checks are recorded in
/// \p Messages (and clear \p IsValid) via checkFailed().
class TargetVerify {
protected:
  void writeValues(ArrayRef<const Value *> Vs) {
    for (const Value *V : Vs) {
      if (!V)
        continue;
      if (isa<Instruction>(V))
        MessagesStr << *V << '\n';
      else {
        V->printAsOperand(MessagesStr, true, Mod);
        MessagesStr << '\n';
      }
    }
  }

  /// A check failed, so print the condition's message.
  ///
  /// This is a convenient place to put a breakpoint to see why something is
  /// considered invalid for the target.
  void checkFailed(const Twine &Message) {
    IsValid = false;
    MessagesStr << Message << '\n';
  }

  /// A check failed (with values to print).
  template <typename T1, typename... Ts>
  void checkFailed(const Twine &Message, const T1 &V1, const Ts &...Vs) {
    checkFailed(Message);
    writeValues({V1, Vs...});
  }

public:
  Module *Mod;
  Triple TT;

  std::string Messages;
  raw_string_ostream MessagesStr;

  /// Whether every function verified so far is valid for the target.
  bool IsValid = true;

  TargetVerify(Module *Mod)
      : Mod(Mod), TT(Mod->getTargetTriple()), MessagesStr(Messages) {}
  virtual ~TargetVerify() = default;

  /// Run this target's checks on \p F. Returns true if \p F is valid.
  virtual bool run(Function &F) = 0;
};

/// Factory that creates a target's TargetVerify for a given module.
using TargetVerifyFactory = TargetVerify *(*)(Module &);

/// Register the TargetVerify factory for \p Arch. Backends call this from their
/// LLVMInitialize<Target>Target() so that the target-independent
/// TargetVerifierPass can dispatch to them by triple. Registering twice for the
/// same arch overwrites the previous factory.
LLVM_ABI void registerTargetVerify(Triple::ArchType Arch,
                                   TargetVerifyFactory Factory);

/// \returns the factory registered for \p Arch, or nullptr if none.
LLVM_ABI const TargetVerifyFactory *getTargetVerify(Triple::ArchType Arch);

/// Create a legacy-PM pass that runs the module's registered TargetVerify. This
/// lets the target verifier run inside the (legacy) codegen pipeline, e.g. via
/// `llc -verify-target`, where it complements the generic IR verifier. If \p
/// FatalErrors is set, an invalid module aborts compilation.
LLVM_ABI ModulePass *createTargetVerifierPass(bool FatalErrors = true);

/// Target-independent dispatcher pass.
///
/// Reads the module triple and, if a TargetVerify is registered for that
/// triple's architecture, runs it over every defined function in the module.
/// Safe to schedule from target-independent pipelines: it is a no-op when no
/// verifier is registered for the target.
class TargetVerifierPass : public PassInfoMixin<TargetVerifierPass> {
  bool FatalErrors;

public:
  explicit TargetVerifierPass(bool FatalErrors = false)
      : FatalErrors(FatalErrors) {}

  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TARGET_TARGETVERIFIER_H
