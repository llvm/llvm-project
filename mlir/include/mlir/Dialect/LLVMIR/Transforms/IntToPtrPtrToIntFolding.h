//===- IntToPtrPtrToIntFolding.h - IntToPtr/PtrToInt folding ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a pass that folds inttoptr/ptrtoint operation sequences.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_INTTOPTRPTRTOINTFOLDING_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_INTTOPTRPTRTOINTFOLDING_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;

namespace LLVM {

#define GEN_PASS_DECL_FOLDINTTOPTRPTRTOINTPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

/// Populate patterns that fold inttoptr/ptrtoint op sequences such as:
///
///   * `inttoptr(ptrtoint(x))` -> `x`
///   * `ptrtoint(inttoptr(x))` -> `x`
///
/// `addressSpaceBWs` contains the pointer bitwidth for each address space. If
/// the pointer bitwidth information is not available for a specific address
/// space, the folding for that address space is not performed.
///
/// TODO: Support DLTI.
void populateIntToPtrPtrToIntFoldingPatterns(
    RewritePatternSet &patterns, ArrayRef<unsigned> addressSpaceBWs);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_INTTOPTRPTRTOINTFOLDING_H