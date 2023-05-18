//===- Context.h - PDLL AST Context -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_AST_CONTEXT_H_
#define MLIR_TOOLS_PDLL_AST_CONTEXT_H_

#include "mlir/Support/StorageUniquer.h"
#include "mlir/Tools/PDLL/AST/Diagnostic.h"

namespace mlir {
namespace pdll {
namespace ods {
class Context;
} // namespace ods

namespace ast {
/// This class represents the main context of the PDLL AST. It handles
/// allocating all of the AST constructs, and manages all state necessary for
/// the AST.
class Context {
public:
  explicit Context(ods::Context &odsContext);
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  /// Return the allocator owned by this context.
  llvm::BumpPtrAllocator &getAllocator() { return allocator; }

  /// Return the storage uniquer used for AST types.
  StorageUniquer &getTypeUniquer() { return typeUniquer; }

  /// Return the ODS context used by the AST.
  ods::Context &getODSContext() { return odsContext; }
  const ods::Context &getODSContext() const { return odsContext; }

  /// Return the diagnostic engine of this context.
  DiagnosticEngine &getDiagEngine() { return diagEngine; }

private:
  /// The diagnostic engine of this AST context.
  DiagnosticEngine diagEngine;

  /// The ODS context used by the AST.
  ods::Context &odsContext;

  /// The allocator used for AST nodes, and other entities allocated within the
  /// context.
  llvm::BumpPtrAllocator allocator;

  /// The uniquer used for creating AST types.
  StorageUniquer typeUniquer;
};

} // namespace ast
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_AST_CONTEXT_H_
