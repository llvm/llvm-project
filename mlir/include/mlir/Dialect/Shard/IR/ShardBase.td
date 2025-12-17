//===- ShardBase.td - Shard Dialect ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHARD_IR_SHARDBASE_TD
#define MLIR_DIALECT_SHARD_IR_SHARDBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Shard Dialect
//===----------------------------------------------------------------------===//

def Shard_Dialect : Dialect {
  let name = "shard";
  let cppNamespace = "::mlir::shard";

  let description = [{
    See [Shard dialect documentation](mlir/docs/Dialects/Shard.md).
  }];

  let dependentDialects = [
    "arith::ArithDialect" // For materializeConstant()
  ];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

def Shard_GridAxis : I<16>;
def Shard_GridAxesAttr : DenseArrayAttrBase<"DenseI16ArrayAttr", "int16_t", "i16">;
def Shard_ShardShapeAttr : DenseArrayAttrBase<"DenseI64ArrayAttr", "int64_t", "i64">;

//===----------------------------------------------------------------------===//
// Shard Enums.
//===----------------------------------------------------------------------===//

def Shard_ReductionKind : I32EnumAttr<"ReductionKind",
  "Reduction of an iterator/grid dimension.", [
  I32EnumAttrCase<"Sum", 1, "sum">,
  I32EnumAttrCase<"Max", 2, "max">,
  I32EnumAttrCase<"Min", 3, "min">,
  I32EnumAttrCase<"Product", 4, "product">,
  // Arithmetic mean.
  I32EnumAttrCase<"Average", 5, "average">,
  I32EnumAttrCase<"BitwiseAnd", 6, "bitwise_and">,
  I32EnumAttrCase<"BitwiseOr", 7, "bitwise_or">,
  I32EnumAttrCase<"BitwiseXor", 8, "bitwise_xor">,
  I32EnumAttrCase<"Generic", 100, "generic">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::shard";
}

def Shard_ReductionKindAttr : EnumAttr<Shard_Dialect, Shard_ReductionKind, "partial"> {
  let assemblyFormat = "$value";
}

class Shard_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Shard_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Shard_Sharding : Shard_Type<"Sharding", "sharding"> {
  let summary = "sharding definition";
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Shard Attribute
//===----------------------------------------------------------------------===//

def Shard_GridAxesArrayAttr : AttrDef<Shard_Dialect, "GridAxesArray"> {
  let mnemonic = "axisarray";
  let parameters = (ins ArrayRefParameter<"GridAxesAttr">:$axes);
  let assemblyFormat = "`[` $axes `]`";
  let extraClassDeclaration = [{
    size_t size() const { return getAxes().size(); }
    auto begin() const { return getAxes().begin(); }
    auto end() const { return getAxes().end(); }
  }];
}

#endif // MLIR_DIALECT_SHARD_IR_SHARDBASE_TD
