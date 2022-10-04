//===- TransformTypes.td - Transform dialect types ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"

def Transform_AnyOpType : TypeDef<Transform_Dialect, "AnyOp",
    [DeclareTypeInterfaceMethods<TransformTypeInterface>]> {
  let description = [{
    Transform IR handle that can be associated with a list of arbitrary
    Payload IR operations.
  }];
  let mnemonic = "any_op";
  let assemblyFormat = "";
}

#endif  // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES
