//===- Passes.h - MemRef Patterns and Passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on MemRef operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

class AffineDialect;
class ModuleOp;

namespace func {
namespace arith {
class ArithDialect;
} // namespace arith
class FuncDialect;
} // namespace func
namespace scf {
class SCFDialect;
} // namespace scf
namespace tensor {
class TensorDialect;
} // namespace tensor
namespace vector {
class VectorDialect;
} // namespace vector

namespace memref {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

/// Additional construction for FoldMemrefAliasOps to allow disabling
/// patterns by name, and controlling folding via a callback function.
/// `controlFn(Operation* userOp)` will be passed the user operation of the
/// aliasing op (e.g., a load/store that uses the result of a memref.subview).
std::unique_ptr<Pass> createFoldMemRefAliasOpsPass(
    ArrayRef<StringRef> excludedPatterns,
    function_ref<bool(Operation *)> controlFn = nullptr);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
