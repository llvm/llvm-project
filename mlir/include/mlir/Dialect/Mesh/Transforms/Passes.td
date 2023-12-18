//===-- Passes.td - Mesh transformation definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_MESH_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_MESH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ShardingPropagation
//===----------------------------------------------------------------------===//

def ShardingPropagation : Pass<"sharding-propagation", "mlir::func::FuncOp"> {
  let summary = "sharding propagation";
  let description = [{
    Propagates sharding information throughout the graph. After this pass, each
    of the operations' operands and results is annotated with a `mesh.shard`
    operation, and the operations themselves are added with sharding option
    attributes.
  }];
  let dependentDialects = [
    "mesh::MeshDialect"
  ];
}

#endif // MLIR_DIALECT_MESH_TRANSFORMS_PASSES_TD
