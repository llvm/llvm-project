
//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES_H_

#include <memory>

namespace mlir {

class Pass;

/// Creates an instance of the ExpandAtomic pass.
std::unique_ptr<Pass> createExpandAtomicPass();

} // end namespace mlir

#endif // MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES_H_
