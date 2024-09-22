//===- MeshBase.td - Mesh Dialect --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_IR_MESHBASE_TD
#define MLIR_DIALECT_MESH_IR_MESHBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Mesh Dialect
//===----------------------------------------------------------------------===//

def Mesh_Dialect : Dialect {
  let name = "mesh";
  let cppNamespace = "::mlir::mesh";

  let description = [{
    See [Mesh dialect documentation](mlir/docs/Dialects/Mesh.md).
  }];

  let dependentDialects = [
    "arith::ArithDialect" // For materializeConstant()
  ];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

def Mesh_MeshAxis : I<16>;
def Mesh_MeshAxesAttr : DenseArrayAttrBase<"DenseI16ArrayAttr", "int16_t", "i16">;
def Mesh_ShardShapeAttr : DenseArrayAttrBase<"DenseI64ArrayAttr", "int64_t", "i64">;

//===----------------------------------------------------------------------===//
// Mesh Enums.
//===----------------------------------------------------------------------===//

def Mesh_ReductionKind : I32EnumAttr<"ReductionKind",
  "Reduction of an iterator/mesh dimension.", [
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
  let cppNamespace = "::mlir::mesh";
}

def Mesh_ReductionKindAttr : EnumAttr<Mesh_Dialect, Mesh_ReductionKind, "partial"> {
  let assemblyFormat = "$value";
}

class Mesh_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Mesh_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Mesh_Sharding : Mesh_Type<"Sharding", "sharding"> {
  let summary = "sharding definition";
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Mesh Attribute
//===----------------------------------------------------------------------===//

def Mesh_MeshAxesArrayAttr : AttrDef<Mesh_Dialect, "MeshAxesArray"> {
  let mnemonic = "axisarray";
  let parameters = (ins ArrayRefParameter<"MeshAxesAttr">:$axes);
  let assemblyFormat = "`[` $axes `]`";
  let extraClassDeclaration = [{
    size_t size() const { return getAxes().size(); }
    auto begin() const { return getAxes().begin(); }
    auto end() const { return getAxes().end(); }
  }];
}

#endif // MLIR_DIALECT_MESH_IR_MESHBASE_TD
