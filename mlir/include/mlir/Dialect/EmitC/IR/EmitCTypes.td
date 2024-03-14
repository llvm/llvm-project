//===- EmitCTypes.td - EmitC types -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR EmitC types.
//
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_EMITC_IR_EMITCTYPES
#define MLIR_DIALECT_EMITC_IR_EMITCTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/EmitC/IR/EmitCBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// EmitC type definitions
//===----------------------------------------------------------------------===//

class EmitC_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EmitC_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def EmitC_ArrayType : EmitC_Type<"Array", "array", [ShapedTypeInterface]> {
  let summary = "EmitC array type";

  let description = [{
    An array data type.

    Example:

    ```mlir
    // Array emitted as `int32_t[10]`
    !emitc.array<10xi32>
    // Array emitted as `float[10][20]`
    !emitc.array<10x20xf32>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];
  let extraClassDeclaration = [{
    /// Returns if this type is ranked (always true).
    bool hasRank() const { return true; }

    /// Clone this array type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    ArrayType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                        Type elementType) const;

    static bool isValidElementType(Type type) {
      return type.isIntOrIndexOrFloat() ||
         llvm::isa<PointerType, OpaqueType>(type);
    }
  }];
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def EmitC_OpaqueType : EmitC_Type<"Opaque", "opaque"> {
  let summary = "EmitC opaque type";

  let description = [{
    An opaque data type of which the value gets emitted as is.

    Example:

    ```mlir
    !emitc.opaque<"int">
    !emitc.opaque<"mytype">
    !emitc.opaque<"std::vector<std::string>">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

def EmitC_PointerType : EmitC_Type<"Pointer", "ptr"> {
  let summary = "EmitC pointer type";

  let description = [{
    A pointer data type.

    Example:

    ```mlir
    // Pointer emitted as `int32_t*`
    !emitc.ptr<i32>
    // Pointer emitted as `float*`
    !emitc.ptr<f32>
    // Pointer emitted as `int*`
    !emitc.ptr<!emitc.opaque<"int">>
    ```
  }];

  let parameters = (ins "Type":$pointee);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$pointee), [{
      return $_get(pointee.getContext(), pointee);
    }]>
  ];
  let assemblyFormat = "`<` qualified($pointee) `>`";
}

#endif // MLIR_DIALECT_EMITC_IR_EMITCTYPES
