//===- MlirQueryMain.h - MLIR Query main ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-query for when built as standalone
// binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MLIRQUERYMAIN_H
#define MLIR_TOOLS_MLIRQUERY_MLIRQUERYMAIN_H

#include "mlir/Query/Matcher/Registry.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

class MLIRContext;

LogicalResult
mlirQueryMain(int argc, char **argv, MLIRContext &context,
              const mlir::query::matcher::Registry &matcherRegistry);

} // namespace mlir

#endif // MLIR_TOOLS_MLIRQUERY_MLIRQUERYMAIN_H
