// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(sharding-propagation{traversal=backward}))" %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  mesh.mesh @mesh(shape = 1) {sym_visibility = "private"}
  func.func @test_forward() -> tensor<6x6xi32> {
    %c1_i32 = arith.constant 1 : i32
    // CHECK: tensor.empty()
    %0 = tensor.empty() : tensor<6x6xi32>
    %sharding = mesh.sharding @mesh split_axes = [[0]] : !mesh.sharding
    // CHECK-COUNT-2: mesh.shard
    %sharding_annotated = mesh.shard %0 to %sharding : tensor<6x6xi32>
    %1 = linalg.fill ins(%c1_i32 : i32) outs(%sharding_annotated : tensor<6x6xi32>) -> tensor<6x6xi32>
    // CHECK: tensor.empty()
    // CHECK-NOT: mesh.shard @
    %2 = tensor.empty() : tensor<6x6xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %1
        : tensor<6x6xi32>, tensor<6x6xi32>) outs(%2 : tensor<6x6xi32>) {
    ^bb0(%in: i32, %in_2: i32, %out: i32):
      %9 = arith.addi %in, %in_2 : i32
      linalg.yield %9 : i32
    } -> tensor<6x6xi32>
    // CHECK: return
    return %3 : tensor<6x6xi32>
  }
}
