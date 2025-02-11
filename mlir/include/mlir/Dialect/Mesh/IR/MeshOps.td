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
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Mesh operations.
//===----------------------------------------------------------------------===//

class Mesh_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mesh_Dialect, mnemonic, traits> {
}

def Mesh_MeshOp : Mesh_Op<"mesh", [Symbol]> {
  let summary = "Description of a device/process mesh.";
  let description = [{
    The mesh.mesh operation is a symbol operation that identifies a specific
    mesh. The operation has three attributes:

    1. `sym_name`: This attribute uniquely identifies the name of the mesh.
    This name serves as a symbolic reference to the mesh throughout
    the MLIR module, allowing for consistent referencing and easier debugging.

    2. `shape`: This attribute represents the shape of the device mesh.
    It uses the same notation as a tensor shape. Also allowing for dynamic
    dimensions.
    This flexibility allows for dynamic device assignment or configurations
    where the exact number of devices might not be determined during compile
    time.
    For example `2x?x4`.

    Example:
    ```
    // A device mesh with 3 axes, the total device number is 4 * 8 * 12
    // The dimension sizes are 4, 8, 12 
    mesh.mesh @mesh0(shape = 4x8x12)

    // A device mesh with 2 axes, the total device number is unknown
    // The first dimension size is 4 and the second is unknown
    mesh.mesh @mesh1(shape = 4x?)

    // A device mesh with 2 axes, the total device number is unknown
    // The first dimension size is unknown and the second is 4
    mesh.mesh @mesh2(shape = ?x4)

    // A device mesh with 2 axes, the number of devices along both axes
    // is unknown
    mesh.mesh @mesh3(shape = ?x?)
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DenseI64ArrayAttr:$shape
  );
  let assemblyFormat = [{
    $sym_name `(` `shape` `=` custom<DimensionList>($shape) `)`
      attr-dict
  }];
  let extraClassDeclaration = [{
    int64_t getRank() { return getShape().size(); }
  }];
  let hasVerifier = 1;
}

def Mesh_MeshShapeOp : Mesh_Op<"mesh_shape", [
    Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Get the shape of the mesh.";
  let arguments = (ins
    FlatSymbolRefAttr:$mesh,
    DefaultValuedAttr<Mesh_MeshAxesAttr, "{}">:$axes
  );

  let results = (outs
    Variadic<Index>:$result
  );

  let assemblyFormat = [{
    $mesh (`axes` `=` $axes^)?
    attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::mesh::MeshOp":$mesh)>,
    OpBuilder<(ins "::mlir::mesh::MeshOp":$mesh, "ArrayRef<MeshAxis>":$axes)>,
    OpBuilder<(ins "StringRef":$mesh, "ArrayRef<MeshAxis>":$axes)>
  ];
}

def Mesh_ProcessMultiIndexOp : Mesh_Op<"process_multi_index", [
  Pure,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "Get the multi index of current device along specified mesh axes.";
  let description = [{
    It is used in the SPMD format of IR.
    The `axes` mush be non-negative and less than the total number of mesh axes.
    If the axes are empty then get the index along all axes.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$mesh,
    DefaultValuedAttr<Mesh_MeshAxesAttr, "{}">:$axes
  );
  let results = (outs
    Variadic<Index>:$result
  );
  let assemblyFormat = [{
    `on` $mesh (`axes` `=` $axes^)?
    attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "::mlir::mesh::MeshOp":$mesh)>,
    OpBuilder<(ins "StringRef":$mesh, "ArrayRef<MeshAxis>":$axes)>
  ];
}

def Mesh_ProcessLinearIndexOp : Mesh_Op<"process_linear_index", [
  Pure,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "Get the linear index of the current device.";
  let description = [{
    Example:
    ```
    %idx = mesh.process_linear_index on @mesh : index
    ```
    if `@mesh` has shape `(10, 20, 30)`, a device with multi
    index `(1, 2, 3)` will have linear index `3 + 30*2 + 20*30*1`.
  }];
  let arguments = (ins FlatSymbolRefAttr:$mesh);
  let results = (outs Index:$result);
  let assemblyFormat = "`on` $mesh attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "::mlir::mesh::MeshOp":$mesh)>
  ];
}

def Mesh_NeighborsLinearIndicesOp : Mesh_Op<"neighbors_linear_indices", [
  Pure,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary =
      "For given mesh index get the linear indices of the direct neighbor processes along the given split.";
  let description = [{
    Example:
    ```
    mesh.mesh @mesh0(shape = 10x20x30)
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %idx = mesh.neighbors_linear_indices on @mesh[%c1, %c2, %c3] split_axes = [1] : index
    ```
    The above returns two indices, `633` and `693`, which correspond to the
    index of the previous process `(1, 1, 3)`, and the next process 
    `(1, 3, 3) along the split axis `1`.

    A negative value is returned if there is no neighbor in the respective
    direction along the given `split_axes`.
  }];
  let arguments = (ins FlatSymbolRefAttr:$mesh,
                       Variadic<Index>:$device,
                       Mesh_MeshAxesAttr:$split_axes);
  let results = (outs Index:$neighbor_down, Index:$neighbor_up);
  let assemblyFormat =  [{
      `on` $mesh `[` $device `]`
      `split_axes` `=` $split_axes
      attr-dict `:` type(results)
  }];
}

//===----------------------------------------------------------------------===//
// Sharding operations.
//===----------------------------------------------------------------------===//

def Mesh_ShardingOp : Mesh_Op<"sharding", [
    Pure,
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Define a sharding of a tensor.";
  let description = [{
    The MeshSharding specifies how a tensor is sharded and distributed across the
    process mesh. It is typically used in a `mesh.shard` operation.
    The operation has the follwing attributes and operands:

    1. `mesh`: this attribute is a FlatSymbolRefAttr that refers to the device
    mesh where the distributed tensor is placed. The symbol must resolve to a
    `mesh.mesh` operation.

    2. `split_axes`: is an array composed of int64_t sub-arrays. The outer array's
    maximum size is the `rank` of the related tensor. For the i-th sub-array, if
    its value is [x, y], it indicates that the tensor's i-th dimension is splitted
    along the x and y axes of the device mesh.

    3. [Optional] `partial_axes`: if not empty, this signifies that the tensor is partial
    one along the specified mesh axes. An all-reduce should be applied to obtain
    the complete tensor, with reduction type being specified by `partial_type`.

    4. [Optional] `partial_type`: indicates the reduction type of the possible all-reduce
    op. It has 4 possible values:
    `generic`: is not an allowed value inside a shard attribute.

    5. [Optional] Sizes of halos to be added for each sharded tensor dimension.
    `halo_sizes` is provided as a flattened 1d array of i64s, 2 values for each
    sharded dimension. `halo_sizes = [1, 2]` means that the first sharded dimension
    gets an additional halo of size 1 at the start of the first dimension and a halo
    size is 2 at its end. `halo_sizes = [1, 2, 2, 3]` defines halos for the first 2
    sharded dimensions e.g. the first sharded dimension gets `[1,2]` halos and the
    seconds gets `[2,3]` halos. `?` indicates dynamic halo sizes.
    
    6. [Optional] Offsets for each shard and sharded tensor dimension.
    `sharded_dims_offsets` is provided as a flattened 1d array of i64s. For each
    sharded tensor dimension the offsets (starting index) of all shards in that
    dimension and an additional value for the end of the last shard are provided.
    For a 1d sharding this means that position `i` has the exclusive prefix sum for
    shard `i`, and since only contiguous sharding is supported, its inclusive prefix
    sum is at position 'i+1'.
    
    Assuming a 3d-tensor of shape 32x32x32 with the first 2 dimensions being sharded,
    `sharded_dims_offsets` = [0, 24, 32, 0, 20, 32] means that the first device of
    the device-mesh will get a shard of shape 24x20x32 and the second device will get
    a shard of shape 8x12x32. `?` indicates dynamic shard dimensions.
    
    `halo_sizes` and `sharded_dims_offsets` are mutually exclusive.

    Examples:

    ```
    mesh.mesh @mesh0(shape = 2x2x4)
    mesh.mesh @mesh1d_4(shape = 4)

    // The tensor is fully replicated on @mesh0.
    // Currently, there must be at least one sub-array present in axes, even
    // if it's empty. Otherwise, a parsing error will occur.
    %sharding0 = mesh.sharding @mesh0 split_axes = [[]]

    // The tensor is sharded on the first dimension along axis 0 of @mesh0
    %sharding1 = mesh.sharding @mesh0 split_axes = [[0]]

    // The tensor is sharded on its first dimension along axis 0 of @mesh0 and
    // it is also a partial_sum along mesh axis 1.
    %sharding2 = mesh.sharding @mesh0 split_axes = [[0] split_axes = []] partial = sum[1]

    // The tensor is sharded on its first dimension along axis 0 of @mesh0 and
    // it is also a partial_max along mesh axis 1.
    %sharding3 = mesh.sharding @mesh0 split_axes = [[0]] partial = max[1]

    // Could be used for a mesh.shard op
    %sharded0 = mesh.shard %arg0 to %sharding3 : tensor<4x8xf32>

    // The tensor is sharded on its first dimension along axis 0 of @mesh0 and
    // and it has halo-sizes of 1 and 2 on the sharded dim.
    %halo_sharding = mesh.sharding @mesh0 split_axes = [[0]] halo_sizes = [1, 2]
    %sharded1 = mesh.shard %arg0 to %halo_sharding : tensor<4x8xf32>
    
    // The tensor is sharded on its second dimension along axis 0 of @mesh1d_4
    // and it has pre-defined shard sizes. The shards of the devices will have
    // the following shapes: [4x2, 4x3, 4x4, 4x5]
    %sharding4 = mesh.sharding @mesh1d_4 split_axes = [[], [0]] sharded_dims_offsets = [0, 2, 5, 9, 14]
    %sharded2 = mesh.shard %arg0 to %sharding4 : tensor<4x14xf32>
    ```
  }];
    
  let arguments = (ins
    FlatSymbolRefAttr:$mesh,
    Mesh_MeshAxesArrayAttr:$split_axes,
    OptionalAttr<Mesh_MeshAxesAttr>:$partial_axes,
    OptionalAttr<Mesh_ReductionKindAttr>:$partial_type,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_sharded_dims_offsets,
    Variadic<I64>:$dynamic_sharded_dims_offsets,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_halo_sizes,
    Variadic<I64>:$dynamic_halo_sizes
  );
  let results = (outs
    Mesh_Sharding:$result
  );
  let assemblyFormat = [{
    $mesh
    `split_axes` `=` $split_axes
    (`partial` `=` $partial_type $partial_axes^)?
    (`halo_sizes` `=` custom<DynamicIndexList>($dynamic_halo_sizes, $static_halo_sizes)^)?
    (`sharded_dims_offsets` `=` custom<DynamicIndexList>($dynamic_sharded_dims_offsets, $static_sharded_dims_offsets)^)?
    attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$mesh,
                   "ArrayRef<MeshAxesAttr>":$split_axes,
                   "ArrayRef<MeshAxis>":$partial_axes,
                   "mesh::ReductionKind":$partial_type,
                   CArg<"ArrayRef<int64_t>", "{}">:$static_halo_sizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$static_sharded_dims_offsets)>,
    OpBuilder<(ins "FlatSymbolRefAttr":$mesh,
                   "ArrayRef<MeshAxesAttr>":$split_axes)>,
    OpBuilder<(ins "FlatSymbolRefAttr":$mesh,
                   "ArrayRef<MeshAxesAttr>":$split_axes,
                   "::mlir::ArrayRef<::mlir::OpFoldResult>":$halo_sizes,
                   "::mlir::ArrayRef<::mlir::OpFoldResult>":$sharded_dims_offsets)>,
    OpBuilder<(ins "mlir::mesh::MeshSharding":$from)>
  ];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Mesh_ShardShapeOp : Mesh_Op<"shard_shape", [Pure]> {
  let summary = "Get the shard shape of a given process/device.";
  let description = [{
    The device/process id is a linearized id of the device/process in the mesh.
    This operation might be used during spmdization when the shard shape depends
    on (non-constant) values used in `mesh.sharding`.
  }];
  let arguments = (ins
    DenseI64ArrayAttr:$shape,
    Mesh_Sharding:$sharding,
    Index:$device
  );
  let results = (outs Variadic<Index>:$result);
  let assemblyFormat = [{
      custom<DimensionList>($shape) $sharding $device attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Value":$sharding, "Value":$device)>
  ];
}

def Mesh_ShardOp : Mesh_Op<"shard", [
    Pure,
    AllTypesMatch<["result", "src"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Annotate on how a tensor is sharded across a mesh.";
  let description = [{
    The mesh.shard operation is designed to specify and guide the sharding
    behavior of a tensor value across a mesh topology. This operation has two
    operands and two optional attributes:

    1. `input`: This operand represents the tensor value that needs to be
    annotated for sharding.

    2. `sharding`: This attribute is type of `MeshShardingType`, which is the core data
    structure to represent distribution of a tensor on a mesh. it is typically defiend
    by an `mesh.sharding` operation.

    3. `annotate_for_users`: A unit attribute addressing the scenario when a
    tensor's sharding annotation differs based on its context of use (either as
    a result or an operand). If specified, the sharding pertains to specific
    users of the tensor value, indicating how it should be considered when used
    as an operand in subsequent operations. If not, the sharding applies to the
    operation that defines the tensor value.

    Example:
    ```
    func.func @only_result_annotated(%arg0 : tensor<4x8xf32>) -> () {
      %sharding = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
      %0 = mesh.shard %arg0 to %sharding : tensor<4x8xf32>
      ...
    }

    func.func @only_operand_annotated(%arg0 : tensor<4x8xf32>) -> () {
      %sharding = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
      %0 = mesh.shard %arg0 to %sharding annotate_for_users : tensor<4x8xf32>
      ...
    }
    
    func.func @two_operands_annotated(%arg0 : tensor<4x8xf32>, %arg1 : tensor<16x8xf32>) -> () {
      %sharding = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
      %0 = mesh.shard %arg0 to %sharding annotate_for_users : tensor<4x8xf32>
      %1 = mesh.shard %arg1 to %sharding annotate_for_users : tensor<16x8xf32>
      ...
    }

    // The first mesh.shard op applies to %arg0, the second mesh.shard op
    // applies for the operand of op0, the third mesh.shard op applies for the
    // operand of op2
    func.func @both_result_and_multi_operands_annotated(
        %arg0 : tensor<4x8xf32>) -> () {
      %sharding = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
      %0 = mesh.shard %arg0 to %sharding : tensor<4x8xf32>
      %sharding1 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
      %1 = mesh.shard %0 to %sharding1 annotate_for_users : tensor<4x8xf32>
      %sharding2 = mesh.sharding @mesh0 split_axes = [[2]] : !mesh.sharding
      %2 = mesh.shard %0 to %sharding2 annotate_for_users : tensor<4x8xf32>
      "op0"(%1) : ...
      "op1"(%2) : ...
      ...
    }
    ```

    The following usages are undefined:
    ```
    func.func @annotate_on_same_result_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %sharding1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
      %sharding2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
      %0 = mesh.shard %arg0 to $sharding1 : tensor<4x8xf32>
      %1 = mesh.shard %0 to sharding2 : tensor<4x8xf32>
      ...
    }

    func.func @annotate_on_same_result_same_value_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %sharding1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
      %sharding2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
      %0 = mesh.shard %arg0 to %sharding1 : tensor<4x8xf32>
      %1 = mesh.shard %arg0 to %sharding2 : tensor<4x8xf32>
      ...
    }

    func.func @annotate_on_same_operand_with_different_sharding(
        %arg0 : tensor<4x8xf32>) -> () {
      %sharding1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
      %sharding2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
      %0 = mesh.shard %arg0 to %sharding1 annotate_for_users : tensor<4x8xf32>
      %1 = mesh.shard %0 to %sharding2 annotate_for_users : tensor<4x8xf32>
      ...
    }

    func.func @result_annotated_after_operand(
        %arg0 : tensor<4x8xf32>) -> () {
      %sharding1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
      %sharding2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
      %0 = mesh.shard %arg0 to %sharding1 annotate_for_users : tensor<4x8xf32>
      %1 = mesh.shard %0 to %sharding2 : tensor<4x8xf32>
      ...
    }
    ```
  }];
  let arguments = (ins
    AnyRankedTensor:$src,
    Mesh_Sharding:$sharding,
    UnitAttr:$annotate_for_users
  );
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $src `to` $sharding
      (`annotate_for_users` $annotate_for_users^)?
      attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// collective communication ops
//===----------------------------------------------------------------------===//

class Mesh_CollectiveCommunicationOpBase<
    string mnemonic, list<Trait> traits = []> :
    Mesh_Op<mnemonic,
      !listconcat(traits,
        [
          DeclareOpInterfaceMethods<SymbolUserOpInterface>,
          DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
        ])> {
  dag commonArgs = (ins
    FlatSymbolRefAttr:$mesh,
    DefaultValuedAttr<Mesh_MeshAxesAttr, "{}">:$mesh_axes
  );
}

def Mesh_AllGatherOp : Mesh_CollectiveCommunicationOpBase<"all_gather", [
    Pure,
    SameOperandsAndResultElementType,
    SameOperandsAndResultRank,
  ]> {
  let summary = "All-gather over a device mesh.";
  let description = [{
    Gathers along the `gather_axis` tensor axis.

    Example:
    ```mlir
    mesh.mesh @mesh0(shape = 2x2)
    ...
    %1 = mesh.all_gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1
      : tensor<2x2xi8> -> tensor<2x4xi8>
    ```
    Input:
    ```
                     +-------+-------+
    device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                     |  3  4 |  7  8 |
                     +-------+-------+
    device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                     | 11 12 | 15 16 |
                     +-------+-------+
    ```
    Result:
    ```
    gather tensor
    axis 1
    ------------>
    +-------------+
    |  1  2  5  6 | <- devices (0, 0) and (0, 1)
    |  3  4  7  8 |
    +-------------+
    |  9 10 13 14 | <- devices (1, 0) and (1, 1)
    | 11 12 15 16 |
    +-------------+
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    IndexAttr:$gather_axis
  ));
  let results = (outs
    AnyNon0RankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? `gather_axis` `=` $gather_axis
    attr-dict `:` type($input) `->` type($result)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_AllReduceOp : Mesh_CollectiveCommunicationOpBase<"all_reduce", [
    Pure,
    SameOperandsAndResultShape]> {
  let summary = "All-reduce over a device mesh.";
  let description = [{
    The accumulation element type is specified by the result type and
    it does not need to match the input element type.
    The input element is converted to the result element type before
    performing the reduction.

    Attributes:
    `reduction`: Indicates the reduction method.

    Example:
    ```
    %1 = mesh.all_reduce %0 on @mesh0 mesh_axes = [1, 0] reduction = <max>
      : tensor<3x4xf32> -> tensor<3x4xf64>
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<Mesh_ReductionKindAttr, "::mlir::mesh::ReductionKind::Sum">:$reduction
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? (`reduction` `=` $reduction^)?
    attr-dict `:` type($input) `->` type($result)
  }];
  let hasCanonicalizer = 1;
    let builders = [
    OpBuilder<(ins "Value":$input, "StringRef":$mesh,
      "ArrayRef<MeshAxis>":$meshAxes, "ReductionKind":$reduction)>
  ];
}

def Mesh_AllSliceOp : Mesh_CollectiveCommunicationOpBase<"all_slice", [
    Pure,
    SameOperandsAndResultElementType,
    SameOperandsAndResultRank
  ]> {
  let summary = "All-slice over a device mesh. This is the inverse of all-gather.";
  let description = [{
    Slice along the `slice_axis` tensor axis.
    This operation can be thought of as the inverse of all-gather.
    Technically, it is not required that all processes have the same input tensor.
    Each process will slice a piece of its local tensor based on its in-group device index.
    The operation does not communicate data between devices. 

    Example:
    ```mlir
    mesh.mesh @mesh0(shape = 2x2)
    ...
    %1 = mesh.all_slice %0 on @mesh0 mesh_axes = [1] slice_axis = 1
      : tensor<2x4xi8> -> tensor<2x2xi8>
    ```
    Input:
    ```
    +-------------+
    |  1  2  5  6 | <- devices (0, 0) and (0, 1)
    |  3  4  7  8 |
    +-------------+
    |  9 10 13 14 | <- devices (1, 0) and (1, 1)
    | 11 12 15 16 |
    +-------------+
    ```
    Result:
    ```
    gather tensor
    axis 1
    ------------>
                     +-------+-------+
    device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                     |  3  4 |  7  8 |
                     +-------+-------+
    device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                     | 11 12 | 15 16 |
                     +-------+-------+
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    IndexAttr:$slice_axis
  ));
  let results = (outs
    AnyNon0RankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? `slice_axis` `=` $slice_axis
    attr-dict `:` type($input) `->` type($result)
  }];
  let hasCanonicalizer = 1;
  let builders = [
    OpBuilder<(ins "Value":$input, "MeshOp":$mesh, "ArrayRef<MeshAxis>":$meshAxes, "int64_t":$sliceAxis)>,
    OpBuilder<(ins "Type":$result_type, "Value":$input, "StringRef":$mesh, "ArrayRef<MeshAxis>":$meshAxes, "int64_t":$sliceAxis)>
  ];
}

def Mesh_AllToAllOp : Mesh_CollectiveCommunicationOpBase<"all_to_all", [
    Pure,
    SameOperandsAndResultElementType,
    SameOperandsAndResultRank]> {
  let summary = "All-to-all over a device mesh.";
  let description = [{
    Performs an all-to-all on tensor pieces split along `split_axis`.
    The resulting pieces are concatenated along `concat_axis` on ech device.

    Example:
    ```
    mesh.mesh @mesh0(shape = 3)
    ...
    %1 = mesh.all_to_all %0 on @mesh0 mesh_axes = [0]
      split_axis = 0 concat_axis = 0
      : tensor<3x2xi8> -> tensor<3x2xi8>
    ```
    Input:
    ```
     device  device  device
     (0)     (1)     (2)
    +-------+-------+-------+  | split and concat along
    | 11 12 | 21 22 | 31 32 |  | tensor axis 0
    | 13 14 | 23 24 | 33 34 |  ↓
    | 15 16 | 25 26 | 35 36 |
    +-------+-------+-------+
    ```
    Result:
    ```
     device  device  device
     (0)     (1)     (2)
    +-------+-------+-------+
    | 11 12 | 13 14 | 15 16 |
    | 21 22 | 23 24 | 25 26 |
    | 31 32 | 33 34 | 35 36 |
    +-------+-------+-------+
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    IndexAttr:$split_axis,
    IndexAttr:$concat_axis
  ));
  let results = (outs
    AnyNon0RankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    `split_axis` `=` $split_axis
    `concat_axis` `=` $concat_axis
    attr-dict `:` type($input) `->` type($result)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_BroadcastOp : Mesh_CollectiveCommunicationOpBase<"broadcast", [
    Pure,
    AllShapesMatch<["input", "result"]>,
    AllElementTypesMatch<["input", "result"]>
  ]> {
  let summary = "Broadcast over a device mesh.";
  let description = [{
    Broadcast the tensor on `root` to all devices in each respective group.
    The operation broadcasts along mesh axes `mesh_axes`.
    The `root` device specifies the in-group multi-index that is broadcast to
    all other devices in the group.
    
    Example:
    ```
    mesh.mesh @mesh0(shape = 2x2)

    %1 = mesh.broadcast %0 on @mesh0
      mesh_axes = [0]
      root = [0]
      : (tensor<2xi8>) -> tensor<2xi8>
    ```
    
    Input:
    ```
                     +-------+-------+                   | broadcast
    device (0, 0) -> |  1  2 |  3  4 | <- device (0, 1)  | along axis 0
                     +-------+-------+                   ↓
    device (1, 0) -> |       |       | <- device (1, 1) 
                     +-------+-------+
    ```

    Output:
    ```
                     +-------+-------+
    device (0, 0) -> |  1  2 |  3  4 | <- device (0, 1)
                     +-------+-------+
    device (1, 0) -> |  1  2 |  3  4 | <- device (1, 1)
                     +-------+-------+
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyRankedTensor:$input,
    DenseI64ArrayAttr:$root,
    Variadic<Index>:$root_dynamic
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    `root` `=` custom<DynamicIndexList>($root_dynamic, $root)
    attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_GatherOp : Mesh_CollectiveCommunicationOpBase<"gather", [
    Pure,
    AllRanksMatch<["input", "result"]>,
    AllElementTypesMatch<["input", "result"]>
  ]> {
  let summary = "Gather over a device mesh.";
  let description = [{
    Gathers on device `root` along the `gather_axis` tensor axis.
    `root` specifies the coordinates of a device along `mesh_axes`.
    It uniquely identifies the root device for each device group.
    The result tensor on non-root devices is undefined.
    Using it will result in undefined behavior.

    Example:
    ```mlir
    mesh.mesh @mesh0(shape = 2x2)
    ...
    %1 = mesh.gather %0 on @mesh0 mesh_axes = [1]
      gather_axis = 1 root = [1]
      : (tensor<2x2xi8>) -> tensor<2x4xi8>
    ```
    Input:
    ```
                      gather tensor
                      axis 1
                      ------------>
                     +-------+-------+
    device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                     |  3  4 |  7  8 |
                     +-------+-------+
    device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                     | 11 12 | 15 16 |
                     +-------+-------+
    ```
    Result:
    ```
    +-------------+
    |  1  2  5  6 | <- devices (0, 1)
    |  3  4  7  8 |
    +-------------+
    |  9 10 13 14 | <- devices (1, 1)
    | 11 12 15 16 |
    +-------------+
    ```
    Devices `(0, 0)` and `(1, 0)` have undefined result.
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    IndexAttr:$gather_axis,
    DenseI64ArrayAttr:$root,
    Variadic<Index>:$root_dynamic
  ));
  let results = (outs
    AnyNon0RankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    `gather_axis` `=` $gather_axis
    `root` `=` custom<DynamicIndexList>($root_dynamic, $root)
    attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_RecvOp : Mesh_CollectiveCommunicationOpBase<"recv", [
    AllShapesMatch<["input", "result"]>,
    AllElementTypesMatch<["input", "result"]>
  ]> {
  let summary = "Send over a device mesh.";
  let description = [{
    Receive from a device within a device group.
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    OptionalAttr<DenseI64ArrayAttr>:$source,
    Variadic<Index>:$source_dynamic
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    (`source` `=` custom<DynamicIndexList>($source_dynamic, $source)^)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_ReduceOp : Mesh_CollectiveCommunicationOpBase<"reduce", [
    Pure,
    AllShapesMatch<["input", "result"]>
  ]> {
  let summary = "Reduce over a device mesh.";
  let description = [{
    Reduces on device `root` within each device group.
    `root` specifies the coordinates of a device along `mesh_axes`.
    It uniquely identifies the root device within its device group.
    The accumulation element type is specified by the result type and
    it does not need to match the input element type.
    The input element is converted to the result element type before
    performing the reduction.

    Attributes:
    `reduction`: Indicates the reduction method.

    Example:
    ```
    %1 = mesh.reduce %0 on @mesh0 mesh_axes = [1, 0]
      reduction = <max> root = [2, 3]
      : (tensor<3x4xf32>) -> tensor<3x4xf64>
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<Mesh_ReductionKindAttr, "::mlir::mesh::ReductionKind::Sum">:$reduction,
    DenseI64ArrayAttr:$root,
    Variadic<Index>:$root_dynamic
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    (`reduction` `=` $reduction^)?
    `root` `=` custom<DynamicIndexList>($root_dynamic, $root)
    attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_ReduceScatterOp : Mesh_CollectiveCommunicationOpBase<"reduce_scatter", [
    Pure,
    SameOperandsAndResultRank]> {
  let summary = "Reduce-scatter over a device mesh.";
  let description = [{
    After the reduction, the result is scattered within each device group.
    The tensor is split along `scatter_axis` and the pieces distributed
    across the device group.
    Example:
    ```
    mesh.mesh @mesh0(shape = 2x2)
    ...
    %1 = mesh.reduce_scatter %0 on @mesh0 mesh_axes = [1]
      reduction = <max> scatter_axis = 0
      : tensor<3x4xf32> -> tensor<1x4xf64>
    ```
    Input:
    ```
                              device
                              (0, 1)
                                 ↓
                     +-------+-------+  | scatter tensor
    device (0, 0) -> |  1  2 |  5  6 |  | axis 0
                     |  3  4 |  7  8 |  ↓
                     +-------+-------+
    device (1, 0) -> |  9 10 | 13 14 |
                     | 11 12 | 15 16 |
                     +-------+-------+
                                ↑
                              device
                              (1, 1)
    ```
    Result:
    ```
    +-------+
    |  6  8 | <- devices (0, 0)
    +-------+
    | 10 12 | <- devices (0, 1)
    +-------+
    | 22 24 | <- devices (1, 0)
    +-------+
    | 26 28 | <- devices (1, 1)
    +-------+
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    DefaultValuedAttr<Mesh_ReductionKindAttr, "::mlir::mesh::ReductionKind::Sum">:$reduction,
    IndexAttr:$scatter_axis
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    (`reduction` `=` $reduction^)?
    `scatter_axis` `=` $scatter_axis
    attr-dict `:` type($input) `->` type($result)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_ScatterOp : Mesh_CollectiveCommunicationOpBase<"scatter", [
    Pure,
    AllRanksMatch<["input", "result"]>,
    AllElementTypesMatch<["input", "result"]>
  ]> {
  let summary = "Scatter over a device mesh.";
  let description = [{
    For each device group split the input tensor on the `root` device along
    axis `scatter_axis` and scatter the parts across the group devices.

    Example:
    ```
    mesh.mesh @mesh0(shape = 2x2)
    %1 = mesh.scatter %0 on @mesh0 mesh_axes = [0]
      scatter_axis = 0
      root = [1]
      : (tensor<2x2xi8>) -> tensor<1x2xi8>
    ```

    Input:
    ```
                              device
                              (0, 1)
                                 ↓
                     +-------+-------+  | scatter tensor
    device (0, 0) -> |       |       |  | axis 0
                     |       |       |  ↓
                     +-------+-------+
    device (1, 0) -> |  1  2 |  5  6 |
                     |  3  4 |  7  8 |
                     +-------+-------+
                                ↑
                              device
                              (1, 1)
    ```
    
    Result:
    ```
                              device
                              (0, 1)
                                 ↓
                     +-------+-------+
    device (0, 0) -> |  1  2 |  5  6 |
                     +-------+-------+ 
    device (1, 0) -> |  3  4 |  7  8 |
                     +-------+-------+
                                ↑
                              device
                              (1, 1)
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    IndexAttr:$scatter_axis,
    DenseI64ArrayAttr:$root,
    Variadic<Index>:$root_dynamic
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    `scatter_axis` `=` $scatter_axis
    `root` `=` custom<DynamicIndexList>($root_dynamic, $root)
    attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_SendOp : Mesh_CollectiveCommunicationOpBase<"send", [
    AllShapesMatch<["input", "result"]>,
    AllElementTypesMatch<["input", "result"]>
  ]> {
  let summary = "Send over a device mesh.";
  let description = [{
    Send from one device to another within a device group.
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    DenseI64ArrayAttr:$destination,
    Variadic<Index>:$destination_dynamic
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    `destination` `=` custom<DynamicIndexList>($destination_dynamic, $destination)
    attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_ShiftOp : Mesh_CollectiveCommunicationOpBase<"shift", [
    Pure,
    SameOperandsAndResultElementType,
    SameOperandsAndResultShape
  ]> {
  let summary = "Shift over a device mesh.";
  let description = [{
    Within each device group shift along mesh axis `shift_axis` by an offset
    `offset`.
    The result on devices that do not have a corresponding source is undefined.
    `shift_axis` must be one of `mesh_axes`.
    If the `rotate` attribute is present,
    instead of a shift a rotation is done.

    Example:
    ```
    mesh.mesh @mesh0(shape = 2x4)
    %1 = mesh.shift on @mesh0 mesh_axes = [1]
      shift_axis = 1 offset = 2 rotate
      : tensor<2xi8> -> tensor<2xi8>
    ```

    Input:
    ```
    mesh axis 1
    ----------->

    +----+----+----+----+
    |  1 |  2 |  3 |  4 |
    +----+----+----+----+
    |  5 |  6 |  7 |  8 |
    +----+----+----+----+
    ```

    Result:
    ```
    +----+----+----+----+
    |  3 |  4 |  1 |  2 |
    +----+----+----+----+
    |  7 |  8 |  5 |  6 |
    +----+----+----+----+
    ```
  }];
  let arguments = !con(commonArgs, (ins
    AnyNon0RankedTensor:$input,
    IndexAttr:$shift_axis,
    I64Attr:$offset,
    UnitAttr:$rotate
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    `shift_axis` `=` $shift_axis
    `offset` `=` $offset
    (`rotate` $rotate^)?
    attr-dict `:` type($input) `->` type($result)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_UpdateHaloOp : Mesh_Op<"update_halo", [
  Pure,
  DestinationStyleOpInterface,
  TypesMatchWith<
    "result has same type as destination",
    "result", "destination", "$_self">,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Update halo data.";
  let description = [{
    This operation updates halo regions of shards, e.g. if their sharding
    specified halos and the actual tensor/memref data might have changed
    on the remote devices. Changes might be caused by mutating operations
    and/or if the new halo regions are larger than the existing ones.

    Destination is supposed to be initialized with the local data (not halos).

    Assumes all devices hold tensors with same-sized halo data as specified
    by `source_halo_sizes/static_source_halo_sizes` and
    `destination_halo_sizes/static_destination_halo_sizes` in source shard
    and destination/result shard.

    `split_axes` specifies for each tensor axis along which mesh axes its halo
    data is updated.

  }];
  let arguments = (ins
    AnyTypeOf<[AnyNon0RankedMemRef, AnyNon0RankedTensor]>:$destination,
    FlatSymbolRefAttr:$mesh,
    Mesh_MeshAxesArrayAttr:$split_axes,
    Variadic<I64>:$halo_sizes,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_halo_sizes
  );
  let results = (outs
    AnyTypeOf<[AnyNon0RankedMemRef, AnyNon0RankedTensor]>:$result
  );
  let assemblyFormat = [{
    $destination
    `on` $mesh
    `split_axes` `=` $split_axes
    (`halo_sizes` `=` custom<DynamicIndexList>($halo_sizes, $static_halo_sizes)^)?
    attr-dict `:` type($result)
  }];
  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDestinationMutable(); }
  }];
}
#endif // MLIR_DIALECT_MESH_IR_MESHOPS_TD
