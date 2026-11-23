//===- SCFToControlFlowDialectInterface.td -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOCONTROLFLOW_SCFTOCONTROLFLOWDIALECTINTERFACE
#define MLIR_CONVERSION_SCFTOCONTROLFLOW_SCFTOCONTROLFLOWDIALECTINTERFACE

include "mlir/IR/Interfaces.td"

def SCFToControlFlowDialectInterface
    : DialectInterface<"SCFToControlFlowDialectInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Controls propagation of discardable loop attributes by the SCF-to-CF
    conversion patterns.

    The patterns copy opted-in attributes from `scf.for` and `scf.while` to the
    branch forming the lowered loop's backedge. When serializing `scf.parallel`,
    they first copy these attributes to the innermost generated `scf.for`, which
    in turn propagates them to its backedge.

    This interface does not authorize attribute propagation by other loop
    transformations, such as tiling, fusing or unrolling. Dialects without this
    interface do not opt in to propagation by these conversion patterns.
  }];
  let methods = [
    InterfaceMethod<[{
        Returns whether the SCF-to-CF conversion patterns should copy `attr`
        unchanged to the replacement described above.
      }],
      "bool", "shouldPropagateLoopAttribute",
      (ins "::mlir::NamedAttribute":$attr),
      [{ return false; }]
    >
  ];
}

#endif // MLIR_CONVERSION_SCFTOCONTROLFLOW_SCFTOCONTROLFLOWDIALECTINTERFACE
