//===-- Passes.td - DLTI pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_PASSES
#define MLIR_DIALECT_DLTI_PASSES

include "mlir/Pass/PassBase.td"

def DltiSetTargetSpecsFromTarget: Pass<"dlti-set-target-specs", ""> {
  let summary = "Sets DLTI target specs using a target.";
  let description = [{
    This pass potentially sets the following DLTI target specs in the current
    operation:
    - The data layout.
    - The target system spec.

    Example:

    ```mlir
    // Given the following input:
    builtin.module @module_1 attributes {dlti.target = #my.target} {...}
    // After applying the pass:
    builtin.module @module_1 attributes {
      dlti.target = #my.target,
      dlti.target_system_spec = #my.system_spec,
      dlti.dl_spec = #my.dl_spec
    } {...}
    ```
  }];
  let dependentDialects = [
    "::mlir::DLTIDialect"
  ];
}

#endif // MLIR_DIALECT_DLTI_PASSES
