// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: test_buffer_cast
func @test_buffer_cast(%arg0: tensor<?xi64>, %arg1: tensor<*xi64>) -> (memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>) {
  %0 = memref.buffer_cast %arg0 : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>
  %1 = memref.buffer_cast %arg1 : memref<*xi64, 1>
  return %0, %1 : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>
}

// CHECK-LABEL: func @memref_reinterpret_cast
func @memref_reinterpret_cast(%in: memref<?xf32>)
    -> memref<10x?xf32, offset: ?, strides: [?, 1]> {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %out = memref.reinterpret_cast %in to
           offset: [%c0], sizes: [10, %c10], strides: [%c10, 1]
           : memref<?xf32> to memref<10x?xf32, offset: ?, strides: [?, 1]>
  return %out : memref<10x?xf32, offset: ?, strides: [?, 1]>
}

// CHECK-LABEL: func @memref_reshape(
func @memref_reshape(%unranked: memref<*xf32>, %shape1: memref<1xi32>,
         %shape2: memref<2xi32>, %shape3: memref<?xi32>) -> memref<*xf32> {
  %dyn_vec = memref.reshape %unranked(%shape1)
               : (memref<*xf32>, memref<1xi32>) -> memref<?xf32>
  %dyn_mat = memref.reshape %dyn_vec(%shape2)
               : (memref<?xf32>, memref<2xi32>) -> memref<?x?xf32>
  %new_unranked = memref.reshape %dyn_mat(%shape3)
               : (memref<?x?xf32>, memref<?xi32>) -> memref<*xf32>
  return %new_unranked : memref<*xf32>
}

// CHECK-LABEL: memref.global @memref0 : memref<2xf32>
memref.global @memref0 : memref<2xf32>

// CHECK-LABEL: memref.global constant @memref1 : memref<2xf32> = dense<[0.000000e+00, 1.000000e+00]>
memref.global constant @memref1 : memref<2xf32> = dense<[0.0, 1.0]>

// CHECK-LABEL: memref.global @memref2 : memref<2xf32> = uninitialized
memref.global @memref2 : memref<2xf32>  = uninitialized

// CHECK-LABEL: memref.global "private" @memref3 : memref<2xf32> = uninitialized
memref.global "private" @memref3 : memref<2xf32>  = uninitialized

// CHECK-LABEL: memref.global "private" constant @memref4 : memref<2xf32> = uninitialized
memref.global "private" constant @memref4 : memref<2xf32>  = uninitialized

// CHECK-LABEL: func @write_global_memref
func @write_global_memref() {
  %0 = memref.get_global @memref0 : memref<2xf32>
  %1 = constant dense<[1.0, 2.0]> : tensor<2xf32>
  memref.tensor_store %1, %0 : memref<2xf32>
  return
}

// CHECK-LABEL: func @read_global_memref
func @read_global_memref() {
  %0 = memref.get_global @memref0 : memref<2xf32>
  %1 = memref.tensor_load %0 : memref<2xf32>
  return
}

// CHECK-LABEL: func @memref_clone
func @memref_clone() {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  %2 = memref.clone %1 : memref<*xf32> to memref<*xf32>
  return
}

// CHECK-LABEL: func @memref_dealloc
func @memref_dealloc() {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  memref.dealloc %1 : memref<*xf32>
  return
}
