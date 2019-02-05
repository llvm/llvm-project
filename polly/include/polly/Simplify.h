//===------ Simplify.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simplify a SCoP by removing unnecessary statements and accesses.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_TRANSFORM_SIMPLIFY_H
#define POLLY_TRANSFORM_SIMPLIFY_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {
class PassRegistry;
class Pass;
} // namespace llvm

namespace polly {

class MemoryAccess;
class ScopStmt;

/// Return a vector that contains MemoryAccesses in the order in
/// which they are executed.
///
/// The order is:
/// - Implicit reads (BlockGenerator::generateScalarLoads)
/// - Explicit reads and writes (BlockGenerator::generateArrayLoad,
///   BlockGenerator::generateArrayStore)
///   - In block statements, the accesses are in order in which their
///     instructions are executed.
///   - In region statements, that order of execution is not predictable at
///     compile-time.
/// - Implicit writes (BlockGenerator::generateScalarStores)
///   The order in which implicit writes are executed relative to each other is
///   undefined.
llvm::SmallVector<MemoryAccess *, 32> getAccessesInOrder(ScopStmt &Stmt);

/// Create a Simplify pass
///
/// @param CallNo Disambiguates this instance for when there are multiple
///               instances of this pass in the pass manager. It is used only to
///               keep the statistics apart and has no influence on the
///               simplification itself.
///
/// @return The Simplify pass.
llvm::Pass *createSimplifyPass(int CallNo = 0);
} // namespace polly

namespace llvm {
void initializeSimplifyPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_TRANSFORM_SIMPLIFY_H */
