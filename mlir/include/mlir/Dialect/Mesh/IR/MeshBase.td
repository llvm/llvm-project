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
    The `mesh` dialect contains a set of attributes, operations, interfaces that
    are useful for representing sharding and communication on device mesh
    cluster.
  }];

  let dependentDialects = [
    "arith::ArithDialect" // For materializeConstant()
  ];

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
}
//===----------------------------------------------------------------------===//
// Mesh Enums.
//===----------------------------------------------------------------------===//

def Mesh_Partial : I32EnumAttr<"Partial", "partial type of a distributed tensor", [
  I32EnumAttrCase<"Sum", 1, "sum">,
  I32EnumAttrCase<"Max", 2, "max">,
  I32EnumAttrCase<"Min", 3, "min">,
  I32EnumAttrCase<"Generic", 100, "generic">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mesh";
}

// Mesh_IteratorType and Mesh_Partial are used to annotate different aspects of
// distributed tensors. Mesh_IteratorType annotates loops in an operation, while
// Mesh_Partial indicates whether a tensor is sharded on a specific dimension or
// is partial.
def Mesh_IteratorType : I32EnumAttr<"IteratorType", "Iterator type", [
  I32EnumAttrCase<"Parallel", 1, "parallel">,
  I32EnumAttrCase<"ReductionSum", 2, "reduction_sum">,
  I32EnumAttrCase<"ReductionMax", 3, "reduction_max">,
  I32EnumAttrCase<"ReductionMin", 4, "reduction_min">,
  I32EnumAttrCase<"ReductionGeneric", 5, "reduction_generic">,
  I32EnumAttrCase<"Invalid", 100, "invalid">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::mesh";
}

//===----------------------------------------------------------------------===//
// Mesh Attribute
//===----------------------------------------------------------------------===//

def MeshSharding : AttrDef<Mesh_Dialect, "MeshSharding"> {
  let mnemonic = "shard";

  let parameters = (ins
    AttrParameter<"::mlir::SymbolRefAttr", "cluster placed">:$cluster,
    ArrayRefParameter<"::mlir::DenseI32ArrayAttr">:$split_axes,
    OptionalArrayRefParameter<"int32_t">:$partial_axes,
    OptionalParameter<"::mlir::mesh::Partial">:$partial_type
  );

  let summary = "Attribute that extends tensor type to distributed tensor type.";

  let description = [{
    The MeshSharding attribute could be used in the encoding of a
    `RankedTensorType` or the mesh.shard op. it contains three sub-attributes:

    1. `cluster`: this attribute is a SymbolRefAttr that refers to the mesh
    cluster where the distributed tensor is placed. The symbol must resolve to a
    `mesh.cluster` operation.

    2. `split_axes`: is an array composed of int64_t sub-arrays. The outer array's
    maximum size is the `rank` of the related tensor. For the i-th sub-array, if
    its value is [x, y], it indicates that the tensor's i-th dimension is splitted
    along the x and y axes of the device mesh.

    3. `partial_axes`: if not empty, this signifies that the tensor is partial
    one along the specified mesh axes. An all-reduce should be applied to obtain
    the complete tensor, with reduction type being specified by `partial_type`.

    4. `partial_type`: indicates the reduction type of the possible all-reduce
    op. It has 4 possible values:
    - `partial_sum`: denotes it's an all-reduce-sum
    - `partial_max`: denotes it's an all-reduce-max
    - `partial_min`: denotes it's an all-reduce-min
    - `partial_generic`: denotes that the all-reduce type is complex and cannot
    be represented merely by a simple sum, max, or min. The exact reduction
    computation may be derived from the semantics of the corresponding operation
    or from the reduction computation IR

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
    // it is also a partial_sum along mesh axis 1.
    tensor<4x8xf32, #mesh.shard<@mesh0, [[0], [], [1]]>

    // The tensor is sharded on the first dimension along axis 0 of @mesh0 and
    // it is also a partial_max along mesh axis 1.
    tensor<4x8xf32, #mesh.shard<@mesh0, [[0]], partial = max[1]>

    // Could be used in the attribute of mesh.shard op
    %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
    ```
  }];
  let assemblyFormat = [{
    `<` $cluster `,` `[` $split_axes `]` (`,` `partial` `=` $partial_type `[`
       $partial_axes^ `]`)? `>`
  }];

  let builders = [
    AttrBuilder<(ins "SymbolRefAttr":$cluster, 
                     "ArrayRef<SmallVector<int32_t>>":$split_axes,
                     "ArrayRef<int32_t>": $partial_axes,
                     "mesh::Partial": $partial_type), [{
      SmallVector<DenseI32ArrayAttr> splitAxesAttr = llvm::map_to_vector(
                  split_axes, [&](ArrayRef<int32_t> array) {
          return DenseI32ArrayAttr::get($_ctxt, array);
      });
      return $_get($_ctxt, cluster, splitAxesAttr, partial_axes,
                   partial_type);
    }]>,
    AttrBuilder<(ins "SymbolRefAttr":$cluster, 
                     "ArrayRef<SmallVector<int32_t>>":$split_axes), [{
      return MeshShardingAttr::get($_ctxt, cluster, split_axes, {}, Partial::Sum);
    }]>
  ];

  let genVerifyDecl = 1;
}

#endif // MLIR_DIALECT_MESH_IR_MESHBASE_TD
