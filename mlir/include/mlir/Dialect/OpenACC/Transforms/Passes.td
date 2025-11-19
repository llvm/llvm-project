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

def ACCImplicitData : Pass<"acc-implicit-data", "mlir::ModuleOp"> {
  let summary = "Generate implicit data attributes for OpenACC compute constructs";
  let description = [{
    This pass implements the OpenACC specification for "Variables with
    Implicitly Determined Data Attributes" (OpenACC 3.4 spec, section 2.6.2).

    The pass automatically generates data clause operations for variables used
    within OpenACC compute constructs (parallel, kernels, serial) that do not
    already have explicit data clauses. The semantics follow these rules:

    1. If there is a default(none) clause visible, no implicit data actions
       apply.

    2. An aggregate variable (arrays, derived types, etc.) will be treated as:
       - In a present clause when default(present) is visible.
       - In a copy clause otherwise.

    3. A scalar variable will be treated as if it appears in:
       - A copy clause if the compute construct is a kernels construct.
       - A firstprivate clause otherwise (parallel, serial).
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect",
      "mlir::memref::MemRefDialect",
      "mlir::arith::ArithDialect"];
  let options = [
    Option<"enableImplicitReductionCopy", "enable-implicit-reduction-copy",
           "bool", "true",
           "Enable applying implicit copy in lieu of implicit firstprivate for "
           "reduction variables. This allows uniform treatment of reduction "
           "variables between combined constructs (e.g., 'parallel loop') and "
           "separate constructs (e.g., 'parallel' followed by 'loop'), where "
           "the OpenACC spec requires copy semantics for the former but "
           "firstprivate would normally apply for the latter.">
  ];
}

def ACCImplicitRoutine : Pass<"acc-implicit-routine", "mlir::ModuleOp"> {
  let summary = "Generate implicit acc routine for functions in acc regions";
  let description = [{
    This pass implements the implicit rules described in OpenACC specification
    for `Routine Directive` (OpenACC 3.4 spec, section 2.15.1).

    "If no explicit routine directive applies to a procedure whose definition
    appears in the program unit being compiled, then the implementation applies
    an implicit routine directive to that procedure if any of the following
    conditions holds:
    - The procedure is called or its address is accessed in a compute region."

    The specification further states:
    "When the implementation applies an implicit routine directive to a procedure,
    it must recursively apply implicit routine directives to other procedures for
    which the above rules specify relevant dependencies. Such dependencies can
    form a cycle, so the implementation must take care to avoid infinite recursion."

    This pass implements these requirements by:
    1. Walking through all OpenACC compute constructs and functions already
       marked with `acc routine` in the module and identifying function calls
       within these regions.
    2. Creating implicit `acc.routine` operations for functions that don't already
       have routine declarations.
    3. Recursively walking through all existing `acc routine` and creating
       implicit routine operations for function calls within these routines,
       while avoiding infinite recursion through proper tracking.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect"];
  let options = [
    Option<"deviceType", "device-type", "mlir::acc::DeviceType",
           "mlir::acc::DeviceType::None",
           "Target device type for implicit routine generation. "
           "Ensures that `acc routine` device_type clauses are "
           "properly considered not just default clauses.">
  ];
}

#endif // MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES
