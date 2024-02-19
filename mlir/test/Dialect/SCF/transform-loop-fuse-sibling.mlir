// RUN: mlir-opt %s -transform-interpreter --cse --canonicalize -split-input-file -verify-diagnostics | FileCheck %s

func.func @test(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: scf.forall ([[I:%.*]]) in (4) shared_outs([[S1:%.*]] = [[IN1:%.*]], [[S2:%.*]] = [[IN2:%.*]]) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  // CHECK:   [[T:%.*]] = affine.apply
  // CHECK:   tensor.extract_slice [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT1:%.*]] = linalg.matmul
  // CHECK:   tensor.extract_slice [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT2:%.*]] = linalg.matmul
  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice [[OUT1]] into [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:     tensor.parallel_insert_slice [[OUT2]] into [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   }
  // CHECK: }
  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A, %B2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop1 into %loop2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @test(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // expected-error @below {{user of results of target should be properly dominated by source}}
  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A, %out1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop1 into %loop2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @test(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  // expected-error @below {{values used inside regions of target should be properly dominated by source}}
  %out2 = linalg.matmul ins(%A, %out1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop2 into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @test(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  // expected-error @below {{operands of target should be properly dominated by source}}
  %out2 = linalg.matmul ins(%A, %B2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out1 : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop2 into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
