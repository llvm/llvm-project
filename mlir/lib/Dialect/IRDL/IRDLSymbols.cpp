//===- IRDLSymbols.cpp - IRDL-related symbol logic --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IRDLSymbols.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"

using namespace mlir;
using namespace mlir::irdl;

static Operation *lookupDialectOp(Operation *source) {
  Operation *dialectOp = source;
  while (dialectOp && !isa<DialectOp>(dialectOp))
    dialectOp = dialectOp->getParentOp();

  if (!dialectOp)
    llvm_unreachable("symbol lookup near dialect must originate from "
                     "within a dialect definition");

  return dialectOp;
}

Operation *
mlir::irdl::lookupSymbolNearDialect(SymbolTableCollection &symbolTable,
                                    Operation *source, SymbolRefAttr symbol) {
  return symbolTable.lookupNearestSymbolFrom(
      lookupDialectOp(source)->getParentOp(), symbol);
}

Operation *mlir::irdl::lookupSymbolNearDialect(Operation *source,
                                               SymbolRefAttr symbol) {
  return SymbolTable::lookupNearestSymbolFrom(
      lookupDialectOp(source)->getParentOp(), symbol);
}
