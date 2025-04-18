//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility function to fix convergence control tokens in the presence of
// irreducible control flow.
//
//===----------------------------------------------------------------------===//

namespace llvm {
class Function;

// Detect and fix invalid convergence control tokens after the entire function
// is emitted in LLVM IR.
void fixConvergenceControl(llvm::Function *F);

} // namespace llvm
