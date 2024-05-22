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
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
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

    3. `dim_sizes`: This attribute represents the shape of the device cluster.
    It uses the same notation as a tensor shape. Also allowing for dynamic
    dimensions.
    This flexibility allows for dynamic device assignment or configurations
    where the exact number of devices might not be determined during compile
    time.
    For example `2x?x4`.

    Example:
    ```
    // A device mesh cluster with 3 axes, the total device number is 4 * 8 * 12
    // The dimension sizes are 4, 8, 12 
    mesh.cluster @mesh0(rank = 3, dim_sizes = 4x8x12)

    // A device mesh cluster with 2 axes, the total device number is unknown
    // The first dimension size is 4 and the second is unknown
    mesh.cluster @mesh1(rank = 2, dim_sizes = 4)

    // A device mesh cluster with 2 axes, the total device number is unknown
    // The first dimension size is unknown and the second is 4
    mesh.cluster @mesh2(rank = 2, dim_sizes = ?x4)

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
    I64Attr:$rank,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$dim_sizes
  );
  let assemblyFormat = [{
    $sym_name `(` `rank` `=` $rank (`,` `dim_sizes` `=` custom<DimensionList>($dim_sizes)^)? `)`
      attr-dict
  }];
  let extraClassDeclaration = [{
    // The `dim_sizes` attribute may have size less than the rank of the mesh.
    // Returns the shape of the mesh with missing trailing dimensions
    // explicitly set as dynamic.
    ::mlir::SmallVector<int64_t> canonicalDimSizes();

    template <typename OutIt>
    void canonicalDimSizes(OutIt outIt) {
      std::copy(getDimSizes().begin(), getDimSizes().end(), outIt);
      std::fill_n(outIt, getRank() - getDimSizes().size(), ::mlir::ShapedType::kDynamic);
    }
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

//===----------------------------------------------------------------------===//
// collective communication ops
//===----------------------------------------------------------------------===//

class Mesh_CollectiveCommunicationOpBase<
    string mnemonic, list<Trait> traits = []> :
    Mesh_Op<mnemonic,
      !listconcat(traits,
      [DeclareOpInterfaceMethods<SymbolUserOpInterface>])> {
  dag commonArgs = (ins
    FlatSymbolRefAttr:$mesh,
    DefaultValuedAttr<DenseI16ArrayAttr, "{}">:$mesh_axes
  );
}

def Mesh_AllGatherOp : Mesh_CollectiveCommunicationOpBase<"all_gather", [
    SameOperandsAndResultElementType,
    SameOperandsAndResultRank
  ]> {
  let summary = "All-gather over a device mesh.";
  let description = [{
    Gathers along the `gather_axis` tensor axis.

    Example:
    ```mlir
    mesh.cluster @mesh0(rank = 2, dim_sizes = 2x2)
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
    DefaultValuedAttr<Mesh_PartialAttr, "::mlir::mesh::Partial::Sum">:$reduction
  ));
  let results = (outs
    AnyRankedTensor:$result
  );
  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? (`reduction` `=` $reduction^)?
    attr-dict `:` type($input) `->` type($result)
  }];
  let hasCanonicalizer = 1;
}

def Mesh_AllToAllOp : Mesh_CollectiveCommunicationOpBase<"all_to_all", [
    SameOperandsAndResultElementType,
    SameOperandsAndResultRank]> {
  let summary = "All-to-all over a device mesh.";
  let description = [{
    Performs an all-to-all on tensor pieces split along `split_axis`.
    The resulting pieces are concatenated along `concat_axis` on ech device.

    Example:
    ```
    mesh.cluster @mesh0(rank = 1, dim_sizes = 3)
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
    mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 2])

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
    mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 2])
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
    DefaultValuedAttr<Mesh_PartialAttr, "::mlir::mesh::Partial::Sum">:$reduction,
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
    SameOperandsAndResultRank]> {
  let summary = "Reduce-scatter over a device mesh.";
  let description = [{
    After the reduction, the result is scattered within each device group.
    The tensor is split along `scatter_axis` and the pieces distributed
    across the device group.
    Example:
    ```
    mesh.cluster @mesh0(rank = 1, dim_sizes = 2x2)
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
    DefaultValuedAttr<Mesh_PartialAttr, "::mlir::mesh::Partial::Sum">:$reduction,
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
    AllRanksMatch<["input", "result"]>,
    AllElementTypesMatch<["input", "result"]>
  ]> {
  let summary = "Scatter over a device mesh.";
  let description = [{
    For each device group split the input tensor on the `root` device along
    axis `scatter_axis` and scatter the parts across the group devices.

    Example:
    ```
    mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 2])
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
    mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])
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

#endif // MLIR_DIALECT_MESH_IR_MESHOPS_TD
