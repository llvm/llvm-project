// RUN: mlir-opt %s -generate-runtime-verification \
// RUN: -one-shot-bufferize="bufferize-function-boundaries" \
// RUN: -buffer-deallocation-pipeline \
// RUN: -convert-bufferization-to-memref \
// RUN: -convert-linalg-to-loops \
// RUN: -expand-strided-metadata \
// RUN: -lower-affine \
// RUN: -convert-scf-to-cf \
// RUN: -test-cf-assert \
// RUN: -convert-index-to-llvm \
// RUN: -finalize-memref-to-llvm \
// RUN: -convert-func-to-llvm \
// RUN: -convert-arith-to-llvm \
// RUN: -convert-cf-to-llvm \
// RUN: -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils \
// RUN:     -shared-libs=%mlir_c_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @main() {
  %c5x = arith.constant dense<0.0> : tensor<5xf32>
  %c4x = arith.constant dense<0.0> : tensor<4xf32>
  %d5x = tensor.cast %c5x : tensor<5xf32> to tensor<?xf32>
  %d4x = tensor.cast %c4x : tensor<4xf32> to tensor<?xf32>

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @simple_add(%d5x, %d5x) : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)

  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.generic
  // CHECK: ^ dimension #0 of input/output operand #1 is incompatible with inferred dimension size
  func.call @simple_add(%d5x, %d4x) : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)

  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.generic
  // CHECK: ^ dimension #0 of input/output operand #1 is incompatible with inferred dimension size
  func.call @simple_add(%d4x, %d5x) : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)

  %c1x1 = arith.constant dense<0.0> : tensor<1x1xf32>
  %c1x4 = arith.constant dense<0.0> : tensor<1x4xf32>
  %c4x4 = arith.constant dense<0.0> : tensor<4x4xf32>
  %c4x5 = arith.constant dense<0.0> : tensor<4x5xf32>
  %c5x4 = arith.constant dense<0.0> : tensor<5x4xf32>
  %d1x1 = tensor.cast %c1x1 : tensor<1x1xf32> to tensor<?x?xf32>
  %d1x4 = tensor.cast %c1x4 : tensor<1x4xf32> to tensor<?x?xf32>
  %d4x4 = tensor.cast %c4x4 : tensor<4x4xf32> to tensor<?x?xf32>
  %d4x5 = tensor.cast %c4x5 : tensor<4x5xf32> to tensor<?x?xf32>
  %d5x4 = tensor.cast %c5x4 : tensor<5x4xf32> to tensor<?x?xf32>

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @broadcast_add(%d1x1, %d1x1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @broadcast_add(%d1x1, %d4x5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @broadcast_add(%d4x4, %d1x4) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.generic
  // CHECK: ^ dimension #1 of input/output operand #1 is incompatible with inferred dimension size
  func.call @broadcast_add(%d1x4, %d4x5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.generic
  // CHECK: ^ dimension #0 of input/output operand #1 is incompatible with inferred dimension size
  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.generic
  // CHECK: ^ dimension #1 of input/output operand #1 is incompatible with inferred dimension size
  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.generic
  // CHECK: ^ dimension #1 of input/output operand #2 is incompatible with inferred dimension size
  func.call @broadcast_add(%d5x4, %d4x5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @matmul_generic(%d5x4, %d4x5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.generic
  // CHECK: ^ dimension #0 of input/output operand #1 is incompatible with inferred dimension size
  func.call @matmul_generic(%d4x5, %d4x5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @matmul_named(%d5x4, %d4x5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.matmul
  // CHECK: ^ dimension #0 of input/output operand #1 is incompatible with inferred dimension size
  func.call @matmul_named(%d4x5, %d4x5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)

  %c64x57 = arith.constant dense<0.0> : tensor<16x29xf32>
  %c3x4 = arith.constant dense<0.0> : tensor<3x4xf32>

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @conv(%c64x57, %c3x4) : (tensor<16x29xf32>, tensor<3x4xf32>) -> (tensor<5x7xf32>)

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @reverse_from_3(%d4x) : (tensor<?xf32>) -> (tensor<?xf32>)

  // CHECK: ERROR: Runtime op verification failed
  // CHECK: linalg.generic
  // CHECK: unexpected negative result on dimension #0 of input/output operand #0
  func.call @reverse_from_3(%d5x) : (tensor<?xf32>) -> (tensor<?xf32>)

  return
}


#identity1D = affine_map<(d0) -> (d0)>

func.func @simple_add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {
    %0 = linalg.generic {
      indexing_maps = [#identity1D, #identity1D],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<?xf32>)
      outs(%arg1 : tensor<?xf32>) {
      ^bb0(%gen_arg1: f32, %gen_arg2: f32) :
        %tmp1 = arith.addf %gen_arg1, %gen_arg2 : f32
        linalg.yield %tmp1 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
}

#broadcastD0 = affine_map<(d0, d1) -> (0, d1)>
#broadcastD1 = affine_map<(d0, d1) -> (d0, 0)>
#identity2D = affine_map<(d0, d1) -> (d0, d1)>

func.func @broadcast_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
 // Calculate maximum dimension 0
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %0 = arith.maxui %dim, %dim_0 : index

  // Calculate maximum dimension 1
  %c1 = arith.constant 1 : index
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dim_2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %1 = arith.maxui %dim_1, %dim_2 : index

  // Broadcast dimension 0 of %arg0
  %dim_3 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %2 = arith.cmpi eq, %dim_3, %c1 : index
  %3 = scf.if %2 -> (tensor<?x?xf32>) {
    %dim_7 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %12 = tensor.empty(%0, %dim_7) : tensor<?x?xf32>
    %13 = linalg.generic {
      indexing_maps = [#broadcastD0, #identity2D],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    scf.yield %13 : tensor<?x?xf32>
  } else {
    scf.yield %arg0 : tensor<?x?xf32>
  }

  // Broadcast dimension 1 of %arg0
  %dim_4 = tensor.dim %3, %c1 : tensor<?x?xf32>
  %4 = arith.cmpi eq, %dim_4, %c1 : index
  %5 = scf.if %4 -> (tensor<?x?xf32>) {
    %dim_7 = tensor.dim %3, %c0 : tensor<?x?xf32>
    %12 = tensor.empty(%dim_7, %1) : tensor<?x?xf32>
    %13 = linalg.generic {
      indexing_maps = [#broadcastD1, #identity2D],
      iterator_types = ["parallel", "parallel"]
    } ins(%3 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    scf.yield %13 : tensor<?x?xf32>
  } else {
    scf.yield %3 : tensor<?x?xf32>
  }

  // Broadcast dimension 0 of %arg1
  %dim_5 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %6 = arith.cmpi eq, %dim_5, %c1 : index
  %7 = scf.if %6 -> (tensor<?x?xf32>) {
    %dim_7 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %12 = tensor.empty(%0, %dim_7) : tensor<?x?xf32>
    %13 = linalg.generic {
      indexing_maps = [#broadcastD0, #identity2D],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg1 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    scf.yield %13 : tensor<?x?xf32>
  } else {
    scf.yield %arg1 : tensor<?x?xf32>
  }

  // Broadcast dimension 1 of %arg1
  %dim_6 = tensor.dim %7, %c1 : tensor<?x?xf32>
  %8 = arith.cmpi eq, %dim_6, %c1 : index
  %9 = scf.if %8 -> (tensor<?x?xf32>) {
    %dim_7 = tensor.dim %7, %c0 : tensor<?x?xf32>
    %12 = tensor.empty(%dim_7, %1) : tensor<?x?xf32>
    %13 = linalg.generic {
      indexing_maps = [#broadcastD1, #identity2D],
      iterator_types = ["parallel", "parallel"]
    } ins(%7 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    scf.yield %13 : tensor<?x?xf32>
  } else {
    scf.yield %7 : tensor<?x?xf32>
  }

  // Perform element-wise computation
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.generic {
    indexing_maps = [#identity2D, #identity2D, #identity2D],
    iterator_types = ["parallel", "parallel"]
  } ins(%5, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_7: f32, %out: f32):
    %12 = arith.addf %in, %in_7 : f32
    linalg.yield %12 : f32
  } -> tensor<?x?xf32>
  return %11 : tensor<?x?xf32>
}

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_trait = {
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = #matmul_accesses
}

func.func @matmul_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cf0 = arith.constant 0.0 : f32 
  %ci0 = arith.constant 0 : index 
  %ci1 = arith.constant 1 : index 
  %d0 = tensor.dim %arg0, %ci0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %ci1 : tensor<?x?xf32>
  %splat = tensor.splat %cf0[%d0, %d1] : tensor<?x?xf32>
  %0 = linalg.generic #matmul_trait ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%splat : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

func.func @matmul_named(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cf0 = arith.constant 0.0 : f32 
  %ci0 = arith.constant 0 : index 
  %ci1 = arith.constant 1 : index 
  %d0 = tensor.dim %arg0, %ci0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %ci1 : tensor<?x?xf32>
  %splat = tensor.splat %cf0[%d0, %d1] : tensor<?x?xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%splat : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

#conv_trait = {
  indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0 * 3 + d2, d1 * 4 + d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction", "reduction"]
}

func.func @conv(%arg0: tensor<16x29xf32>, %arg1: tensor<3x4xf32>) -> (tensor<5x7xf32>) {
  %c0 = arith.constant 0.0 : f32 
  %splat = tensor.splat %c0 : tensor<5x7xf32>
  %result = linalg.generic #conv_trait ins(%arg0, %arg1 : tensor<16x29xf32>, tensor<3x4xf32>) outs(%splat : tensor<5x7xf32>) {
  ^bb0(%in: f32, %in_64: f32, %out: f32):
    %5 = arith.mulf %in, %in_64 : f32
    %6 = arith.addf %out, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<5x7xf32>
  return %result : tensor<5x7xf32>
}

#reverse_trait = {
  indexing_maps = [
          affine_map<(i) -> (3 - i)>,
          affine_map<(i) -> (i)>
  ],
  iterator_types = ["parallel"]
}

func.func @reverse_from_3(%arg0: tensor<?xf32>) -> (tensor<?xf32>) {
  %cf0 = arith.constant 0.0 : f32 
  %ci0 = arith.constant 0 : index 
  %d0 = tensor.dim %arg0, %ci0 : tensor<?xf32>
  %splat = tensor.splat %cf0[%d0] : tensor<?xf32>
  %result = linalg.generic #reverse_trait ins(%arg0: tensor<?xf32>) outs(%splat: tensor<?xf32>) {
    ^bb0(%a: f32, %b: f32):
    linalg.yield %a : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
