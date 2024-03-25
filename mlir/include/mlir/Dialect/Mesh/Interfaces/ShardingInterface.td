//===- ShardingInterfaces.td -------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_TD
#define MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_TD

include "mlir/IR/OpBase.td"

def ShardingInterface : OpInterface<"ShardingInterface"> {
    let description = [{
        Interface for allowing operations to expose information needed to
        shard them.
    }];
    let cppNamespace = "::mlir::mesh";

    let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of iterator types that describe the number of loops.
          The iterator types determine how the operation traverses its input and
          output tensors.

          Example 1: A gemm op has 3 loops, M, N and K. Their loop iterator
          types are parallel, parallel, reduction. This indicates that M and
          N are traversed in parallel, while the K dimension is used for
          reduction.
        }],
        /*retType=*/"SmallVector<mlir::utils::IteratorType>",
        /*methodName=*/"getLoopIteratorTypes",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return {};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the kind of all reduction loop iterators.
          The order is the same as the same as the result from
          `getLoopIteratorTypes`.

          Example 1:
          iterator types =  (parallel, reduction, parallel, reduction)
                                             ||                   ||
          reduction kinds = (                sum,                 max)

          Example 2:
          A softmax op's loop iterator types are parallel and
          reduction.
          The reduction iterator will be of kind `generic`, since it is non of
          the available presets.
        }],
        /*retType=*/"SmallVector<ReductionKind>",
        /*methodName=*/"getReductionLoopIteratorKinds",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return {};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the indexing maps attribute within the current operation.
          Indexing maps determine how indices in the iteration space map to
          tensor indices. They are specified using `affine_map` in MLIR, which
          provides an affine transformation of indices.
        }],
        /*retTy=*/"SmallVector<AffineMap>",
        /*methodName=*/"getIndexingMaps",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return {};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Given that certain operands or results of the operation may have
          sharding annotations, this method leverages this information to deduce
          how the operation should be sharded.
        }],
        /*retTy=*/"FailureOr<ShardingOption>",
        /*methodName=*/"getShardingOption",
        /*args=*/(ins
          "ArrayRef<MeshShardingAttr>": $operandShardings,
          "ArrayRef<MeshShardingAttr>": $resultShardings
        ),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return detail::defaultGetShardingOption(
            $_op.getOperation(), operandShardings, resultShardings);
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Based on a given ShardingOption, this method adds `mesh.shard`
          operations for the operands and results that previously lacked
          sharding annotations.
        }],
        /*retTy=*/"LogicalResult",
        /*methodName=*/"addShardingAnnotations",
        /*args=*/(ins
          "OpBuilder &":$b,
          "const ShardingOption &":$shardingOption
        ),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return detail::defaultAddShardingAnnotations(
            $_op.getOperation(), b, shardingOption);
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Convert self to SPMD form.
          This method is used during the spmdization pass of a program fully
          annotated with shardings.

          The spmdization algorithm would read the surrounding sharding
          annotations from the IR for each argument/result and prepare
          `operandShardings` and `resultShardings`.
          Values that are not ranked tensors do not have sharding annotations.
          In this case their corresponding MeshShardingAttr is null.

          For convenience it will also prepare `spmdizedOperands`, although
          they can be retrieved from the `spmdizationMap`.

          The `spmdizationMap` contains a mapping from unsharded to
          sharded/spmdized values that are constructed during the spmdization
          pass. The interface implementation must populate `spmdizationMap`
          with the mapping for this op's results.

          `builder` is set to insert new operations in the appropriate point.
          The implementation should not return the builder to the original
          insertion point.
          It should leave it as is after all insertions are done.

          The default implementation does full replication.
          This assumes that all sharding annotations are for full replication.
        }],
        /*retTy=*/"LogicalResult",
        /*methodName=*/"spmdize",
        /*args=*/(ins
          "ArrayRef<Value>": $spmdizedOperands,
          "ArrayRef<MeshShardingAttr>": $operandShardings,
          "ArrayRef<MeshShardingAttr>": $resultShardings,
          "IRMapping&": $spmdizationMap,
          "SymbolTableCollection &": $symbolTableCollection,
          "OpBuilder &":$builder
        ),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          spmdizeFullyReplicatedOperation(
            *$_op.getOperation(), spmdizedOperands, operandShardings,
              resultShardings, spmdizationMap, symbolTableCollection, builder);
          return success();
        }]>
    ];

    let extraClassDeclaration = [{
      LogicalResult verifyShardingInterfaceImpl();

      void printLoopTypesAndIndexingMaps(raw_ostream &os);
    }];
}


#endif // MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_TD
