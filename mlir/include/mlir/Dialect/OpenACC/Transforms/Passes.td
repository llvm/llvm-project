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

def LegalizeDataValuesInRegion : Pass<"openacc-legalize-data-values", "mlir::func::FuncOp"> {
  let summary = "Legalizes SSA values in compute regions with results from data clause operations";
  let description = [{
    This pass replace uses of the `varPtr` in compute regions (kernels,
    parallel, serial) with the result of data clause operations (`accPtr`).
  }];
  let options = [
    Option<"hostToDevice", "host-to-device", "bool", "true",
           "Replace varPtr uses with accPtr if true. Replace accPtr uses with "
           "varPtr if false">,
    Option<"applyToAccDataConstruct", "apply-to-acc-data-construct", "bool", "true",
           "Replaces varPtr uses with accPtr for acc compute regions contained "
           "within acc.data or acc.declare region.">
  ];
}

#endif // MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES
