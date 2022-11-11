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

/// Returns true if the given operation is free of memory effects.
///
/// An operation is free of memory effects if its implementation of
/// `MemoryEffectOpInterface` indicates that it has no memory effects. For
/// example, it may implement `NoMemoryEffect` in ODS. Alternatively, if the
/// operation has the `HasRecursiveMemoryEffects` trait, then it is free of
/// memory effects if all of its nested operations are free of memory effects.
///
/// If the operation has both, then it is free of memory effects if both
/// conditions are satisfied.
bool isMemoryEffectFree(Operation *op);

/// Returns true if the given operation is speculatable, i.e. has no undefined
/// behavior or other side effects.
///
/// An operation can indicate that it is speculatable by implementing the
/// getSpeculatability hook in the ConditionallySpeculatable op interface.
bool isSpeculatable(Operation *op);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_SIDEFFECTUTILS_H
