//===- SMTExtensionOps.td - Transform dialect operations ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSIONOPS
#define MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSIONOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstrainParamsOp : Op<Transform_Dialect, "smt.constrain_params", [
  DeclareOpInterfaceMethods<TransformOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  NoTerminator
]> {
  let cppNamespace = [{ mlir::transform::smt }];

  let summary = "Express contraints on params interpreted as symbolic values";
  let description = [{
    Allows expressing constraints on params using the SMT dialect.

    Each Transform dialect param provided as an operand has a corresponding
    argument of SMT-type in the region. The SMT-Dialect ops in the region use
    these arguments as operands.

    The semantics of this op is that all the ops in the region together express
    a constraint on the params-interpreted-as-smt-vars. The op fails in case the
    expressed constraint is not satisfiable per SMTLIB semantics. Otherwise the
    op succeeds.

    ---

    TODO: currently the operational semantics per the Transform interpreter is
    to always fail. The intention is build out support for hooking in your own
    operational semantics so you can invoke your favourite solver to determine
    satisfiability of the corresponding constraint problem.
  }];

  let arguments = (ins Variadic<TransformParamTypeInterface>:$params);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "`(` $params `)` attr-dict `:` type(operands) $body";

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSIONOPS
