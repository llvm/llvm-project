//===- SideEffectUtils.h - Side Effect Utils --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SIDEFFECTUTILS_H
#define MLIR_TRANSFORMS_SIDEFFECTUTILS_H

namespace mlir {

class Operation;

/// Returns true if the given operation is side-effect free.
///
/// An operation is side-effect free if its implementation of
/// `MemoryEffectOpInterface` indicates that it has no memory effects. For
/// example, it may implement `NoSideEffect` in ODS. Alternatively, if the
/// operation `HasRecursiveSideEffects`, then it is side-effect free if all of
/// its nested operations are side-effect free.
///
/// If the operation has both, then it is side-effect free if both conditions
/// are satisfied.
bool isSideEffectFree(Operation *op);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_SIDEFFECTUTILS_H
