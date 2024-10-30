//===- mlir-link.cpp - MLIR linker ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  mlir-link a.mlir b.mlir c.mlir -o x.mlir
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-link/MlirLinkMain.h"

using namespace mlir;

int main(int argc, char **argv) {
    DialectRegistry registry;
    registerAllDialects(registry);
    registerAllExtensions(registry);

    return failed(MlirLinkMain(argc, argv, registry));
}
