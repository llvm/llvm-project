//===- Argument.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_ARGUMENT_H
#define LLVM_SANDBOXIR_ARGUMENT_H

#include "llvm/IR/Argument.h"
#include "llvm/SandboxIR/Value.h"

namespace llvm::sandboxir {

/// Argument of a sandboxir::Function.
class Argument : public sandboxir::Value {
  Argument(llvm::Argument *Arg, sandboxir::Context &Ctx)
      : Value(ClassID::Argument, Arg, Ctx) {}
  friend class Context; // For constructor.

public:
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Argument;
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Argument>(Val) && "Expected Argument!");
  }
  void printAsOperand(raw_ostream &OS) const;
  void dumpOS(raw_ostream &OS) const final;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_ARGUMENT_H
