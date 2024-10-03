//===- Pass.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_PASS_H
#define LLVM_SANDBOXIR_PASS_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::sandboxir {

class Function;

/// The base class of a Sandbox IR Pass.
class Pass {
protected:
  /// The pass name. This is also used as a command-line flag and should not
  /// contain whitespaces.
  const std::string Name;

public:
  Pass(StringRef Name) : Name(Name) {
    assert(!Name.contains(' ') &&
           "A pass name should not contain whitespaces!");
    assert(!Name.starts_with('-') && "A pass name should not start with '-'!");
  }
  virtual ~Pass() {}
  /// \Returns the name of the pass.
  StringRef getName() const { return Name; }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const Pass &Pass) {
    Pass.print(OS);
    return OS;
  }
  virtual void print(raw_ostream &OS) const { OS << Name; }
  LLVM_DUMP_METHOD virtual void dump() const;
#endif
  /// Similar to print() but adds a newline. Used for testing.
  void printPipeline(raw_ostream &OS) const { OS << Name << "\n"; }
};

/// A pass that runs on a sandbox::Function.
class FunctionPass : public Pass {
public:
  FunctionPass(StringRef Name) : Pass(Name) {}
  /// \Returns true if it modifies \p F.
  virtual bool runOnFunction(Function &F) = 0;
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_PASS_H
