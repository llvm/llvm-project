//===- PDLTypes.td - Pattern descriptor types --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Pattern Descriptor Language dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDL_IR_PDLTYPES
#define MLIR_DIALECT_PDL_IR_PDLTYPES

include "mlir/Dialect/PDL/IR/PDLDialect.td"

//===----------------------------------------------------------------------===//
// PDL Types
//===----------------------------------------------------------------------===//

class PDL_Type<string name, string typeMnemonic>
    : TypeDef<PDL_Dialect, name, [], "::mlir::pdl::PDLType"> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// pdl::AttributeType
//===----------------------------------------------------------------------===//

def PDL_Attribute : PDL_Type<"Attribute", "attribute"> {
  let summary = "PDL handle to an `mlir::Attribute`";
  let description = [{
    This type represents a handle to an instance of an `mlir::Attribute`, bound
    to a value that is usable within a PDL pattern or rewrite.
  }];
}

//===----------------------------------------------------------------------===//
// pdl::OperationType
//===----------------------------------------------------------------------===//

def PDL_Operation : PDL_Type<"Operation", "operation"> {
  let summary = "PDL handle to an `mlir::Operation *`";
  let description = [{
    This type represents a handle to an instance of an `mlir::Operation *`,
    bound to a value that is usable within a PDL pattern or rewrite.
  }];
}

//===----------------------------------------------------------------------===//
// pdl::RangeType
//===----------------------------------------------------------------------===//

def PDL_Range : PDL_Type<"Range", "range"> {
  let summary = "PDL handle to a range of a given sub-type";
  let description = [{
    This type represents a range of instances of the given PDL element type,
    i.e. `Attribute`, `Operation`, `Type`, or `Value`.
  }];
  let parameters = (ins "Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
  ];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// pdl::TypeType
//===----------------------------------------------------------------------===//

def PDL_Type : PDL_Type<"Type", "type"> {
  let summary = "PDL handle to an `mlir::Type`";
  let description = [{
    This type represents a handle to an instance of an `mlir::Type`, bound to a
    value that is usable within a PDL pattern or rewrite.
  }];
}

//===----------------------------------------------------------------------===//
// pdl::ValueType
//===----------------------------------------------------------------------===//

def PDL_Value : PDL_Type<"Value", "value"> {
  let summary = "PDL handle for an `mlir::Value`";
  let description = [{
    This type represents a handle to an instance of an `mlir::Value`, bound to a
    value that is usable within a PDL pattern or rewrite.
  }];
}

//===----------------------------------------------------------------------===//
// Additional Type Constraints
//===----------------------------------------------------------------------===//

def PDL_AnyType : Type<
  CPred<"$_self.isa<::mlir::pdl::PDLType>()">, "pdl type",
        "::mlir::pdl::PDLType">;

// A range of positional values of one of the provided types.
class PDL_RangeOf<Type positionalType> :
  ContainerType<AnyTypeOf<[positionalType]>, PDL_Range.predicate,
                "$_self.cast<::mlir::pdl::RangeType>().getElementType()",
                "range", "::mlir::pdl::RangeType">,
    BuildableType<"::mlir::pdl::RangeType::get(" # positionalType.builderCall #
                  ")">;

// Either a positional value or a range of positional values for a given type.
class PDL_InstOrRangeOf<Type positionalType> :
    AnyTypeOf<[positionalType, PDL_RangeOf<positionalType>],
              "single element or range of " # positionalType.summary,
              "::mlir::pdl::PDLType">;

#endif // MLIR_DIALECT_PDL_IR_PDLTYPES
