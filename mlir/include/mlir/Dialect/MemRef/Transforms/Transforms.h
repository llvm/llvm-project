//===- Transforms.h - MemRef Dialect transformations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This header declares functions that assit transformations in the MemRef
/// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_TRANSFORMS_H

namespace mlir {
class RewritePatternSet;

namespace memref {
/// Appends patterns for extracting address computations from the instructions
/// with memory accesses such that these memory accesses use only a base
/// pointer.
///
/// For instance,
/// ```mlir
/// memref.load %base[%off0, ...]
/// ```
///
/// Will be rewritten in:
/// ```mlir
/// %new_base = memref.subview %base[%off0,...][1,...][1,...]
/// memref.load %new_base[%c0,...]
/// ```
void populateExtractAddressComputationsPatterns(RewritePatternSet &patterns);

} // namespace memref
} // namespace mlir

#endif
