// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: test_index_cast
func @test_index_cast(%arg0 : index) -> i64 {
  %0 = index_cast %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: test_index_cast_tensor
func @test_index_cast_tensor(%arg0 : tensor<index>) -> tensor<i64> {
  %0 = index_cast %arg0 : tensor<index> to tensor<i64>
  return %0 : tensor<i64>
}

// CHECK-LABEL: test_index_cast_tensor_reverse
func @test_index_cast_tensor_reverse(%arg0 : tensor<i64>) -> tensor<index> {
  %0 = index_cast %arg0 : tensor<i64> to tensor<index>
  return %0 : tensor<index>
}

// CHECK-LABEL: test_tensor_to_memref
func @test_tensor_to_memref(%arg0: tensor<?xi64>, %arg1: tensor<*xi64>) -> (memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>) {
  %0 = tensor_to_memref %arg0 : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>
  %1 = tensor_to_memref %arg1 : memref<*xi64, 1>
  return %0, %1 : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>
}

// CHECK-LABEL: @assert
func @assert(%arg : i1) {
  assert %arg, "Some message in case this assertion fails."
  return
}

// CHECK-LABEL: @dynamic_tensor_from_elements
func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = constant 8.0 : f32
      yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// CHECK-LABEL: @atan
func @atan(%arg : f32) -> f32 {
  %result = atan %arg : f32
  return %result : f32
}

// CHECK-LABEL: @atan2
func @atan2(%arg0 : f32, %arg1 : f32) -> f32 {
  %result = atan2 %arg0, %arg1 : f32
  return %result : f32
}

// CHECK-LABEL: func @memref_reshape(
func @memref_reshape(%unranked: memref<*xf32>, %shape1: memref<1xi32>,
         %shape2: memref<2xi32>, %shape3: memref<?xi32>) -> memref<*xf32> {
  %dyn_vec = memref_reshape %unranked(%shape1)
               : (memref<*xf32>, memref<1xi32>) -> memref<?xf32>
  %dyn_mat = memref_reshape %dyn_vec(%shape2)
               : (memref<?xf32>, memref<2xi32>) -> memref<?x?xf32>
  %new_unranked = memref_reshape %dyn_mat(%shape3)
               : (memref<?x?xf32>, memref<?xi32>) -> memref<*xf32>
  return %new_unranked : memref<*xf32>
}
