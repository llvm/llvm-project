//===-- MeshOps.td - Mesh dialect operation definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_IR_MESHOPS_TD
#define MLIR_DIALECT_MESH_IR_MESHOPS_TD

include "mlir/Dialect/Mesh/IR/MeshBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Mesh Dialect operations.
//===----------------------------------------------------------------------===//

class Mesh_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mesh_Dialect, mnemonic, traits> {
}

def Mesh_ClusterOp : Mesh_Op<"cluster", [Symbol]> {
  let summary = "representing a mesh cluster";
  let description = [{
    The mesh.cluster operation is a symbol operation that identifies a specific
    mesh cluster. The operation has three attributes:

    1. `sym_name`: This attribute uniquely identifies the name of the mesh
    cluster. This name serves as a symbolic reference to the cluster throughout
    the MLIR module, allowing for consistent referencing and easier debugging.

    2. `rank`: This attribute specifies the number of axes of the cluster. The
    rank indicates the dimensionality of the mesh cluster and can be used to
    determine the layout and the addressing space of the computation distributed
    across the mesh.

    3. `dim_sizes`: This attribute represents the device assignment along the
    axes of the cluster. Each integer in the array corresponds to the number of
    devices along a specific axis. If an integer value is 0, it implies that the
    number of devices along that axis is unknown. This flexibility allows for
    dynamic device assignment or configurations where the exact number of
    devices might not be determined during compile time.

    Example:
    ```
    // A device mesh cluster with 3 axes, the total device number is 4 * 8 * 12
    // The dimension sizes are 4, 8, 12 
    mesh.cluster @mesh0(rank = 3, dim_sizes = [4, 8, 12])

    // A device mesh cluster with 2 axes, the total device number is unknown
    // The first dimension size is 4 and the second is unknown
    mesh.cluster @mesh1(rank = 2, dim_sizes = [4])

    // A device mesh cluster with 2 axes, the total device number is unknown
    // The first dimension size is unknown and the second is 4
    mesh.cluster @mesh2(rank = 2, dim_sizes = [0, 4])

    // A device mesh cluster with 2 axes, the number of devices along both axes
    // is unknown
    mesh.cluster @mesh3(rank = 2)

    // Used in the mesh sharding attribute to extend the standard tensor to
    // distributed
    tensor<4x8xf32, #mesh.shard<@mesh0, [[0]]>>
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I8Attr:$rank,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$dim_sizes
  );
  let assemblyFormat = [{
    $sym_name `(` `rank` `=` $rank (`,` `dim_sizes` `=` $dim_sizes^)? `)`
      attr-dict
  }];
  let hasVerifier = 1;
}

def Mesh_ShardOp : Mesh_Op<"shard", [Pure, SameOperandsAndResultType]> {
  let summary = "Annotate on how a tensor is sharded across a mesh cluster.";
  let description = [{
    The mesh.shard operation is designed to specify and guide the sharding
    behavior of a tensor value across a mesh topology. This operation has one
    operand and two attributes:

    1. `input`: This operand represents the tensor value that needs to be
    annotated for sharding.

    2. `shard`: This attribute is type of `MeshSharding`, which is the core data
    structure to represent distributed tensor in mesh cluster.

    3. `annotate_for_users`: A unit attribute addressing the scenario when a
    tensor's sharding annotation differs based on its context of use (either as
    a result or an operand). If specified, the sharding pertains to specific
    users of the tensor value, indicating how it should be considered when used
    as an operand in subsequent operations. If not, the sharding applies to the
    operation that defines the tensor value.

    Example:
    ```
    func.func @only_result_annotated(%arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
      ...
    }

    func.func @only_operand_annotated(%arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
      ...
    }

    // The first mesh.shard op applies to %arg0, the second mesh.shard op
    // applies for the operand of op0, the third mesh.shard op applies for the
    // operand of op2
    func.func @both_result_and_multi_operands_annotated(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
      %1 = mesh.shard %0 to <@mesh0, [[1]]> annotate_for_users : tensor<4x8xf32>
      %2 = mesh.shard %0 to <@mesh0, [[2]]> annotate_for_users : tensor<4x8xf32>
      "op0"(%1) : ...
      "op1"(%2) : ...
      ...
    }
    ```

    The following usages are undefined:
    ```
    func.func @annotate_on_same_result_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
      %1 = mesh.shard %0 to <@mesh0, [[1]]> : tensor<4x8xf32>
      ...
    }

    func.func @annotate_on_same_result_same_value_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
      %1 = mesh.shard %arg0 to <@mesh0, [[1]]> : tensor<4x8xf32>
      ...
    }

    func.func @annotate_on_same_operand_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
      %1 = mesh.shard %0 to <@mesh0, [[1]]> annotate_for_users : tensor<4x8xf32>
      ...
    }

    func.func @result_annotated_after_operand(
        %arg0 : tensor<4x8xf32>) -> () {
      %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
      %1 = mesh.shard %0 to <@mesh0, [[1]]> : tensor<4x8xf32>
      ...
    }
    ```
  }];
  let arguments = (ins
    Builtin_RankedTensor:$src,
    MeshSharding:$shard,
    UnitAttr:$annotate_for_users
  );
  let results = (outs
    Builtin_RankedTensor:$result
  );
  let assemblyFormat = [{
    $src `to` $shard (`annotate_for_users` $annotate_for_users^)? attr-dict `:`
      type($result)
  }];
}

#endif // MLIR_DIALECT_MESH_IR_MESHOPS_TD
