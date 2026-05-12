// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(sharding-propagation{traversal=forward}))" %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "mpich", "MPI:comm_world_rank" = 0 : i32>} {
  shard.grid @grid(shape = 1) {sym_visibility = "private"}
  func.func @test_forward() -> (tensor<6x6xi32>, tensor<6x6xi32>, tensor<i32>) attributes {llvm.emit_c_interface} {
    %c1_i32 = arith.constant 1 : i32
    // CHECK: [[v3:%.*]] = tensor.empty() : tensor<6x6xi32>
    %0 = tensor.empty() : tensor<6x6xi32>
    // CHECK: [[v1:%.*]] = linalg.fill ins
    // CHECK: [[vsharding_0:%.*]] = shard.sharding @grid split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK: [[vsharded_1:%.*]] = shard.shard [[v1]] to [[vsharding_0]] : tensor<6x6xi32>
    %1 = linalg.fill ins(%c1_i32 : i32) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
    %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
    %sharded = shard.shard %1 to %sharding : tensor<6x6xi32>
    // CHECK: [[v2:%.*]] = tensor.empty() : tensor<6x6xi32>
    // CHECK: [[vsharding_2:%.*]] = shard.sharding @grid split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK: [[vsharded_3:%.*]] = shard.shard [[vsharded_1]] to [[vsharding_2]] annotate_for_users : tensor<6x6xi32>
    %3 = tensor.empty() : tensor<6x6xi32>
    // CHECK: [[vsharding_4:%.*]] = shard.sharding @grid split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK: [[vsharded_5:%.*]] = shard.shard [[v2]] to [[vsharding_4]] annotate_for_users : tensor<6x6xi32>
    // CHECK: [[v3:%.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
    // CHECK-SAME: ins([[vsharded_3]], [[vsharded_3]] : tensor<6x6xi32>, tensor<6x6xi32>) outs([[vsharded_5]] : tensor<6x6xi32>) {
    // CHECK: [[vsharding_6:%.*]] = shard.sharding @grid split_axes = {{\[\[}}0]] : !shard.sharding
    // CHECK: [[vsharded_7:%.*]] = shard.shard [[v3]] to [[vsharding_6]] : tensor<6x6xi32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%sharded, %sharded
        : tensor<6x6xi32>, tensor<6x6xi32>) outs(%3 : tensor<6x6xi32>) {
    ^bb0(%in: i32, %in_2: i32, %out: i32):
      %9 = arith.addi %in, %in_2 : i32
      linalg.yield %9 : i32
    } -> tensor<6x6xi32>
    %c0_i32 = arith.constant 0 : i32
    %6 = tensor.empty() : tensor<i32>
    %7 = linalg.fill ins(%c0_i32 : i32) outs(%6 : tensor<i32>) -> tensor<i32>
    // CHECK: [[vreduced:%.*]] = linalg.reduce ins
    // CHECK: [[vsharding_12:%.*]] = shard.sharding @grid split_axes = [] : !shard.sharding
    // CHECK: [[vsharded_13:%.*]] = shard.shard [[vreduced]] to [[vsharding_12]] : tensor<i32>
    %reduced = linalg.reduce ins(%4 : tensor<6x6xi32>) outs(%7 : tensor<i32>) dimensions = [0, 1] 
      (%in: i32, %init: i32) {
        %9 = arith.addi %in, %init : i32
        linalg.yield %9 : i32
      }
    // CHECK: [[vsharding_14:%.*]] = shard.sharding @grid split_axes = {{\[\[}}]] : !shard.sharding
    %sharding_0 = shard.sharding @grid split_axes = [[]] : !shard.sharding
    // CHECK: [[vsharded_15:%.*]] = shard.shard [[vsharded_13]] to [[vsharding_14]] annotate_for_users : tensor<i32>
    %sharded_1 = shard.shard %reduced to %sharding_0 annotate_for_users : tensor<i32>
    return %sharded, %4, %sharded_1 : tensor<6x6xi32>, tensor<6x6xi32>, tensor<i32>
  }
}
