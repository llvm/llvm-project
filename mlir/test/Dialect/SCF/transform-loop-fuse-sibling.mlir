// RUN: mlir-opt %s -transform-interpreter --cse --canonicalize -split-input-file -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s --check-prefix CHECK-NOCLEANUP

// CHECK: func.func @fuse_1st_for_into_2nd([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @fuse_1st_for_into_2nd(%A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: [[R0:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] iter_args([[IA:%.*]] = [[A]], [[IB:%.*]] = [[B]]) {{.*}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IA]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IA]][[[IV]]]
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %B) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IB]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IB]][[[IV]]]
    %dup2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#0 into %for#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func.func @fuse_2nd_for_into_1st([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @fuse_2nd_for_into_1st(%A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: [[R0:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] iter_args([[IB:%.*]] = [[B]], [[IA:%.*]] = [[A]]) {{.*}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IB]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IB]][[[IV]]]
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %B) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IA]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IA]][[[IV]]]
    %dup2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
  // NB: the dominance check used to fail on the following line,
  // however the defining op for the value of %arg3 occurs above the source loop and hence is safe
  // and %arg4 is a block argument of the scope of the loops and hence is safe
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#1 into %for#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func.func @matmul_fuse_1st_forall_into_2nd([[A1:%.*]]: {{.*}}, [[A2:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @matmul_fuse_1st_forall_into_2nd(%A1 : tensor<128x128xf32>, %A2 : tensor<128x128xf32>, %B : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: scf.forall ([[I:%.*]]) in (4) shared_outs([[S1:%.*]] = [[IN1:%.*]], [[S2:%.*]] = [[IN2:%.*]]) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  // CHECK:   [[T:%.*]] = affine.apply
  // CHECK:   tensor.extract_slice [[A2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   tensor.extract_slice [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT1:%.*]] = linalg.matmul
  // CHECK:   tensor.extract_slice [[A1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   tensor.extract_slice [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT2:%.*]] = linalg.matmul
  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice [[OUT1]] into [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:     tensor.parallel_insert_slice [[OUT2]] into [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   }
  // CHECK: }
  %out1 = linalg.matmul ins(%A1, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A2, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

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

// CHECK: func.func @matmul_fuse_2nd_forall_into_1st([[A1:%.*]]: {{.*}}, [[A2:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @matmul_fuse_2nd_forall_into_1st(%A1 : tensor<128x128xf32>, %A2 : tensor<128x128xf32>, %B : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: scf.forall ([[I:%.*]]) in (4) shared_outs([[S1:%.*]] = [[IN1:%.*]], [[S2:%.*]] = [[IN2:%.*]]) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  // CHECK:   [[T:%.*]] = affine.apply
  // CHECK:   tensor.extract_slice [[A1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   tensor.extract_slice [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT1:%.*]] = linalg.matmul
  // CHECK:   tensor.extract_slice [[A2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   tensor.extract_slice [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT2:%.*]] = linalg.matmul
  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice [[OUT1]] into [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:     tensor.parallel_insert_slice [[OUT2]] into [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   }
  // CHECK: }
  %out1 = linalg.matmul ins(%A1, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A2, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

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

// CHECK-NOCLEANUP: func.func @fuse_no_iter_args([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @fuse_no_iter_args(%A: tensor<128xf32>, %B: tensor<128xf32>) {
  // CHECK-NOCLEANUP: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-NOCLEANUP: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-NOCLEANUP: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-NOCLEANUP: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NOCLEANUP: scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] {{.*}}
  scf.for %arg0 = %c0 to %c128 step %c16 {
  // CHECK-NOCLEANUP:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
    %2 = vector.transfer_read %A[%arg0], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    scf.yield
  }
  scf.for %arg0 = %c0 to %c128 step %c16 {
  // CHECK-NOCLEANUP:   [[BSLICE:%.*]] = vector.transfer_read [[B]][[[IV]]], [[ZERO]]
    %dup2 = vector.transfer_read %B[%arg0], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    scf.yield
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#0 into %for#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

func.func @source_for_uses_result_of_target_for_err(%A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // expected-error @below {{user of results of target should be properly dominated by source}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %1) -> (tensor<128xf32>) {
    %dup2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#0 into %for#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

func.func @source_forall_uses_result_of_target_forall_err(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
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

func.func @target_for_region_uses_result_of_source_for_err(%A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %B) -> (tensor<128xf32>) {
  // expected-error @below {{values used inside regions of target should be properly dominated by source}}
    %dup2 = vector.transfer_read %1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#1 into %for#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

func.func @target_forall_depends_on_value_not_dominated_by_source_forall_err(%A1 : tensor<128x128xf32>, %A2 : tensor<128x128xf32>, %B : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %buf1_alloc = tensor.empty() : tensor<128x128xf32>
  %buf1 = linalg.fill ins(%zero : f32) outs(%buf1_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out1 = linalg.matmul ins(%A1, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%buf1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out_alloc2 = tensor.empty() : tensor<128x128xf32>
  %buf2 = linalg.fill ins(%zero : f32) outs(%buf1_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>
  // expected-error @below {{operands of target should be properly dominated by source}}
  %out2 = linalg.matmul ins(%A2, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%buf2 : tensor<128x128xf32>) -> tensor<128x128xf32>

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

// CHECK: func.func @foreach_loop_pair_fuse([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @foreach_loop_pair_fuse(%arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: [[RST:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] iter_args([[IB0:%.*]] = [[B]], [[IB1:%.*]] = [[B]]) {{.*}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IB0]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IB0]][[[IV]]]
    %2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  } {target_loops}
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IB1]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IB1]][[[IV]]]
    %dup2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  } {source_loops}
  %2 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128xf32>)  {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IB0]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IB0]][[[IV]]]
    %2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %5 = arith.addf %3, %2 : vector<32xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<32xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  } {target_loops}
  %dup2 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IB1]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IB1]][[[IV]]]
    %dup2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<32xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<32xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  } {source_loops}
  return %1, %dup1, %2, %dup2 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %target_loops = transform.structured.match ops{["scf.for"]} attributes {target_loops} in %arg0 : (!transform.any_op) -> !transform.any_op
    %source_loops = transform.structured.match ops{["scf.for"]} attributes {source_loops} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.foreach %target_loops, %source_loops : !transform.any_op, !transform.any_op {
    ^bb0(%target_loop: !transform.any_op, %source_loop: !transform.any_op):
      %fused = transform.loop.fuse_sibling %target_loop into %source_loop : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    }
    transform.yield
  }
}
