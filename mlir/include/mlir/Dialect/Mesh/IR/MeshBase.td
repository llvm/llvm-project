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

//===----------------------------------------------------------------------===//
// Mesh Dialect
//===----------------------------------------------------------------------===//

def Mesh_Dialect : Dialect {
  let name = "mesh";
  let cppNamespace = "::mlir::mesh";

  let description = [{
    The `mesh` dialect contains a set of attributes, operations, interfaces that
    are useful for representing sharding and communication on device mesh
    cluster.
  }];

  let dependentDialects = [
    "arith::ArithDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Mesh Attribute
//===----------------------------------------------------------------------===//

def MeshSharding : AttrDef<Mesh_Dialect, "MeshSharding"> {
  let mnemonic = "shard";

  let parameters = (ins
    OptionalParameter<"::mlir::SymbolRefAttr">:$cluster,
    ArrayRefParameter<"::mlir::DenseI64ArrayAttr">:$axes
  );

  let summary = "Attribute that extends tensor type to distributed tensor type.";

  let description = [{
    The mesh.shard attribute contains two attribute in it:
    1. `cluster`: this attribute is a SymbolRefAttr that refers to the mesh
    cluster where the distributed tensor is placed.

    2. `axes`: is an array composed of int64_t sub-arrays. The outer array's
    maximum size is the `rank` of the related tensor plus one. For the i-th
    sub-array, if its value is [x, y]:
    - When i < `rank`, it indicates that the tensor's i-th dimension is sharded
    along the x and y axes of the device mesh.
    - When i == `rank`, it signifies that the tensor represents a partial sum
    along the x and y axes. More partial types could be introduced if needed, 
    e.g. partial-max, partial-min and even partial-generic.

    Example:

    ```
    mesh.cluster @mesh0(rank = 3, dim_sizes = [2, 2, 4])

    // The tensor is fully replicated on @mesh0.
    // Currently, there must be at least one sub-array present in axes, even
    // if it's empty. Otherwise, a parsing error will occur.
    tensor<4x8xf32, #mesh.shard<@mesh0, [[]]>>

    // The tensor is sharded on the first dimension along axis 0 of @mesh0
    tensor<4x8xf32, #mesh.shard<@mesh0, [[0]]>

    // The tensor is sharded on the first dimension along axis 0 of @mesh0 and
    // it is also a partial-sum along axis 1.
    tensor<4x8xf32, #mesh.shard<@mesh0, [[0], [], [1]]>

    // Could also be used in the attribute of mesh.shard op
    %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
    ```
  }];
  let assemblyFormat = "`<` ($cluster^ `,`)? `[` $axes `]` `>`";
  let genVerifyDecl = 1;
}

#endif // MLIR_DIALECT_MESH_IR_MESHBASE_TD
