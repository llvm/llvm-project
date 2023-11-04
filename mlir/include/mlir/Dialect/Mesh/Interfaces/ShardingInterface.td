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
          types are parallel, parallel, reduction-sum. This indicates that M and
          N are traversed in parallel, while the K dimension is used for
          reduction.

          Example 2: A softmax op's loop iterator types are parallel and
          invalid. The second dimension is considered as invalid because it is
          neither parallel nor any kind of reduction. 
        }],
        /*retType=*/"SmallVector<::mlir::mesh::IteratorType>",
        /*methodName=*/"getLoopIteratorTypes",
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
      >
    ];

    let extraClassDeclaration = [{
      LogicalResult verifyShardingInterfaceImpl();

      void printLoopTypesAndIndexingMaps(raw_ostream &os);
    }];
}


#endif // MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_TD
