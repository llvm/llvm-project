//===- ConvertSCFToStandard.h - Pass entrypoint -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOSTANDARD_SCFTOSTANDARD_H_
#define MLIR_CONVERSION_SCFTOSTANDARD_SCFTOSTANDARD_H_

#include <memory>
#include <vector>

namespace mlir {
struct LogicalResult;
class Pass;

class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

/// Collect a set of patterns to lower from scf.for, scf.if, and
/// loop.terminator to CFG operations within the Standard dialect, in particular
/// convert structured control flow into CFG branch-based control flow.
void populateLoopToStdConversionPatterns(OwningRewritePatternList &patterns);

/// Creates a pass to convert scf.for, scf.if and loop.terminator ops to CFG.
std::unique_ptr<Pass> createLowerToCFGPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOSTANDARD_SCFTOSTANDARD_H_
