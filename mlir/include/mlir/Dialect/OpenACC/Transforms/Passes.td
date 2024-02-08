//===-- Passes.td - OpenACC pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES
#define MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LegalizeDataInRegion : Pass<"openacc-legalize-data", "mlir::func::FuncOp"> {
  let summary = "Legalize the data in the compute region";
  let description = [{
    This pass replace uses of varPtr in the compute region with their accPtr
    gathered from the data clause operands.
  }];
  let options = [
    Option<"hostToDevice", "host-to-device", "bool", "true",
           "Replace varPtr uses with accPtr if true. Replace accPtr uses with "
           "varPtr if false">
  ];
  let constructor = "::mlir::acc::createLegalizeDataInRegion()";
}

#endif // MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES
