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
          "OpBuilder &":$b
        ),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return detail::defaultGetShardingOption(
            $_op.getOperation(), b);
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
