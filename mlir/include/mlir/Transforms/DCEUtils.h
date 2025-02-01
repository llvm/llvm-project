//===- DCE.h - Dead Code Elimination -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares methods for eliminating dead code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_DCE_H_
#define MLIR_TRANSFORMS_DCE_H_

namespace mlir {

class Operation;
class RewriterBase;

/// Eliminate dead code within the given `target`.
void deadCodeElimination(RewriterBase &rewriter, Operation *target);

} // namespace mlir

#endif // MLIR_TRANSFORMS_DCE_H_
