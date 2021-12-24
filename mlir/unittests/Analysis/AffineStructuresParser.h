//===- AffineStructuresParser.h - Parser for AffineStructures ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helper functions to parse AffineStructures from
// StringRefs.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UNITTEST_ANALYSIS_AFFINESTRUCTURESPARSER_H
#define MLIR_UNITTEST_ANALYSIS_AFFINESTRUCTURESPARSER_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
/// This parses a single IntegerSet to an MLIR context and transforms it to
/// FlatAffineConstraints if it was valid. If not, a failure is returned. If the
/// passed `str` has additional tokens that were not part of the IntegerSet, a
/// failure is returned. Diagnostics are printed on failure if
/// `printDiagnosticInfo` is true.

FailureOr<FlatAffineConstraints>
parseIntegerSetToFAC(llvm::StringRef, MLIRContext *context,
                     bool printDiagnosticInfo = true);

} // namespace mlir

#endif // MLIR_UNITTEST_ANALYSIS_AFFINESTRUCTURESPARSER_H
