//===- AffineStructuresParser.cpp - Parser for AffineStructures -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./AffineStructuresParser.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

FailureOr<FlatAffineConstraints>
mlir::parseIntegerSetToFAC(llvm::StringRef str, MLIRContext *context,
                           bool printDiagnosticInfo) {
  IntegerSet set = parseIntegerSet(str, context, printDiagnosticInfo);

  if (!set)
    return failure();

  return FlatAffineConstraints(set);
}
