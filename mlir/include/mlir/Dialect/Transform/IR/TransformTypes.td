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

def Transform_AffineMapParamType : TypeDef<Transform_Dialect, "AffineMapParam",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let description = [{
    Transform IR parameter value that can be associated with a list of affine
    map attributes.
  }];
  let mnemonic = "affine_map";
  let assemblyFormat = "";
}

def Transform_AnyOpType : TypeDef<Transform_Dialect, "AnyOp",
    [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let description = [{
    Transform IR handle that can be associated with a list of arbitrary
    Payload IR operations.
  }];
  let mnemonic = "any_op";
  let assemblyFormat = "";
}

def Transform_AnyValue : TypeDef<Transform_Dialect, "AnyValue",
    [DeclareTypeInterfaceMethods<TransformValueHandleTypeInterface>]> {
  let description = [{
    Transform IR value that can be associated with a list of Payload IR values.
  }];
  let mnemonic = "any_value";
  let assemblyFormat = "";
}

def Transform_OperationType : TypeDef<Transform_Dialect, "Operation",
    [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let description = [{
    Transform IR handle that can be associated with a list of Payload IR
    operations with the specified operation name.
  }];
  let mnemonic = "op";
  let parameters = (ins
    StringRefParameter<"Name of the allowed payload operation">:$operation_name
  );
  let assemblyFormat = "`<` $operation_name `>`";
}

def Transform_ParamType : TypeDef<Transform_Dialect, "Param",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let description = [{
    Transform IR value that can be associated with the list of parameters
    of the given type. Types are currently limited to integers, but may be
    extended in the future to other types values of which can be contained
    in attributes.
  }];
  let mnemonic = "param";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Underlying type of the parameter">:$type
  );
  let assemblyFormat = "`<` $type `>`";
  let genVerifyDecl = 1;
}

def Transform_TypeParamType : TypeDef<Transform_Dialect, "TypeParam",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let description = [{
    Transform IR parameter value that can be associated with a list of type
    attributes.
  }];
  let mnemonic = "type";
  let assemblyFormat = "";
}

class Transform_ConcreteOpType<string opname>
  : Type<And<[Transform_OperationType.predicate,
              CPred<"::llvm::cast<::mlir::transform::OperationType>($_self)"
                    ".getOperationName() == \"" # opname # "\"">]>,
         "Transform IR handle to " # opname # " operations",
         "::mlir::transform::OperationType">;

def TransformAnyHandle : Type<
    Or<[TransformHandleTypeInterface.predicate,
        TransformValueHandleTypeInterface.predicate]>,
    "transform operation or value handle">;

#endif  // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES
