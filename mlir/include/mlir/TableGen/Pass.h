//===- Pass.h - TableGen pass definitions -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PASS_H_
#define MLIR_TABLEGEN_PASS_H_

#include "mlir/Support/LLVM.h"
#include <vector>

namespace llvm {
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {
//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Wrapper class providing helper methods for Passes defined in TableGen.
class Pass {
public:
  explicit Pass(const llvm::Record *def);

  /// Return the command line argument of the pass.
  StringRef getArgument() const;

  /// Return the short 1-line summary of the pass.
  StringRef getSummary() const;

  /// Return the description of the pass.
  StringRef getDescription() const;

  /// Return the C++ constructor call to create an instance of this pass.
  StringRef getConstructor() const;

private:
  const llvm::Record *def;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_PASS_H_
