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
  let hasConstantMaterializer = 1;
}

def Mesh_MeshAxis : I<16>;
def Mesh_MeshAxesAttr : DenseArrayAttrBase<"DenseI16ArrayAttr", "int16_t", "i16">;

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
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Mesh Attribute
//===----------------------------------------------------------------------===//

def MeshSharding : AttrDef<Mesh_Dialect, "MeshSharding"> {
  let mnemonic = "shard";

  let parameters = (ins
    AttrParameter<"::mlir::FlatSymbolRefAttr",
     "The mesh on which tensors are sharded.">:$mesh,
    ArrayRefParameter<"MeshAxesAttr">:$split_axes,
    OptionalArrayRefParameter<"MeshAxis">:$partial_axes,
    OptionalParameter<"::mlir::mesh::ReductionKind">:$partial_type
  );

  let summary = "Attribute that extends tensor type to distributed tensor type.";

  let description = [{
    The MeshSharding attribute is used in a `mesh.shard` operation.
    It specifies how a tensor is sharded and distributed across the process
    mesh.

    1. `mesh`: this attribute is a FlatSymbolRefAttr that refers to the device
    mesh where the distributed tensor is placed. The symbol must resolve to a
    `mesh.mesh` operation.

    2. `split_axes`: is an array composed of int64_t sub-arrays. The outer array's
    maximum size is the `rank` of the related tensor. For the i-th sub-array, if
    its value is [x, y], it indicates that the tensor's i-th dimension is splitted
    along the x and y axes of the device mesh.

    3. `partial_axes`: if not empty, this signifies that the tensor is partial
    one along the specified mesh axes. An all-reduce should be applied to obtain
    the complete tensor, with reduction type being specified by `partial_type`.

    4. `partial_type`: indicates the reduction type of the possible all-reduce
    op. It has 4 possible values:
    `generic`: is not an allowed value inside a shard attribute.

    Example:

    ```
    mesh.mesh @mesh0(shape = 2x2x4)

    // The tensor is fully replicated on @mesh0.
    // Currently, there must be at least one sub-array present in axes, even
    // if it's empty. Otherwise, a parsing error will occur.
    #mesh.shard<@mesh0, [[]]>

    // The tensor is sharded on the first dimension along axis 0 of @mesh0
    #mesh.shard<@mesh0, [[0]]>

    // The tensor is sharded on the first dimension along axis 0 of @mesh0 and
    // it is also a partial_sum along mesh axis 1.
    #mesh.shard<@mesh0, [[0], []], partial = sum[1]>

    // The tensor is sharded on the first dimension along axis 0 of @mesh0 and
    // it is also a partial_max along mesh axis 1.
    #mesh.shard<@mesh0, [[0]], partial = max[1]>

    // Could be used in the attribute of mesh.shard op
    %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
    ```
  }];
  let assemblyFormat = [{
    `<` $mesh `,` `[` $split_axes `]` (`,` `partial` `=` $partial_type `[`
       $partial_axes^ `]`)? `>`
  }];

  let builders = [
    AttrBuilder<(ins "FlatSymbolRefAttr":$mesh,
                     "ArrayRef<SmallVector<MeshAxis>>":$split_axes,
                     "ArrayRef<MeshAxis>": $partial_axes,
                     "mesh::ReductionKind": $partial_type), [{
      SmallVector<MeshAxesAttr> splitAxesAttr = llvm::map_to_vector(
                  split_axes, [&](ArrayRef<MeshAxis> array) {
          return MeshAxesAttr::get($_ctxt, array);
      });
      return $_get($_ctxt, mesh, splitAxesAttr, partial_axes,
                   partial_type);
    }]>,
    AttrBuilder<(ins "FlatSymbolRefAttr":$mesh,
                     "ArrayRef<SmallVector<MeshAxis>>":$split_axes), [{
      return MeshShardingAttr::get($_ctxt, mesh, split_axes, {}, ReductionKind::Sum);
    }]>
  ];

  let extraClassDeclaration = [{
    bool operator==(::mlir::Attribute rhs) const;
    bool operator!=(::mlir::Attribute rhs) const;
    bool operator==(::mlir::mesh::MeshShardingAttr rhs) const;
    bool operator!=(::mlir::mesh::MeshShardingAttr rhs) const;
  }];

  let genVerifyDecl = 1;
}

#endif // MLIR_DIALECT_MESH_IR_MESHBASE_TD
