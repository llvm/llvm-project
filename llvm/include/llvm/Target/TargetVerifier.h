//===-- llvm/Target/TargetVerifier.h - LLVM IR Target Verifier --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines target verifier interfaces that can be used for some
// validation of input to the system, and for checking that transformations
// haven't done something bad. In contrast to the Verifier or Lint, the
// TargetVerifier looks for constructions invalid to a particular target
// machine.
//
// To see what specifically is checked, look at an individual backend's
// TargetVerifier.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_VERIFIER_H
#define LLVM_TARGET_VERIFIER_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

class Function;

class TargetVerifierPass : public PassInfoMixin<TargetVerifierPass> {
public:
  virtual PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) = 0;
};

class TargetVerify {
protected:
  void writeValues(ArrayRef<const Value *> Vs) {
    for (const Value *V : Vs) {
      if (!V)
        continue;
      if (isa<Instruction>(V)) {
        MessagesStr << *V << '\n';
      } else {
        V->printAsOperand(MessagesStr, true, Mod);
        MessagesStr << '\n';
      }
    }
  }

  /// A check failed, so printout out the condition and the message.
  ///
  /// This provides a nice place to put a breakpoint if you want to see why
  /// something is not correct.
  void checkFailed(const Twine &Message) { MessagesStr << Message << '\n'; }

  /// A check failed (with values to print).
  ///
  /// This calls the Message-only version so that the above is easier to set
  /// a breakpoint on.
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

  bool IsValid = true;

  TargetVerify(Module *Mod)
      : Mod(Mod), TT(Mod->getTargetTriple()), MessagesStr(Messages) {}

  virtual bool run(Function &F) = 0;
};

} // namespace llvm

#endif // LLVM_TARGET_VERIFIER_H
