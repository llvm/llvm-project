// RUN: mlir-opt %s -vector-bufferize -split-input-file | FileCheck %s

// CHECK-LABEL: func @transfer_read(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[o1:.*]]: index, %[[o2:.*]]: index, %[[pad:.*]]: f32)
//       CHECK:   %[[m:.*]] = bufferization.to_memref %[[t]] : memref<?x?xf32>
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[m]][%[[o1]], %[[o2]]], %[[pad]] {in_bounds = [true, false]} : memref<?x?xf32>, vector<5x6xf32>
//       CHECK:   return %[[r]]
func.func @transfer_read(%t: tensor<?x?xf32>, %o1: index,
                    %o2: index, %pad: f32) -> vector<5x6xf32> {
  %0 = vector.transfer_read %t[%o1, %o2], %pad {in_bounds = [true, false]}
      : tensor<?x?xf32>, vector<5x6xf32>
  return %0 : vector<5x6xf32>
}

// -----

// CHECK-LABEL: func @transfer_write(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[o1:.*]]: index, %[[o2:.*]]: index, %[[vec:.*]]: vector<5x6xf32>)
//       CHECK:   %[[m:.*]] = bufferization.to_memref %[[t]] : memref<?x?xf32>
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}}, %{{.*}}) {{.*}} : memref<?x?xf32>
//       CHECK:   memref.copy %[[m]], %[[alloc]]
//       CHECK:   vector.transfer_write %[[vec]], %[[alloc]][%[[o1]], %[[o2]]] {in_bounds = [true, false]} : vector<5x6xf32>, memref<?x?xf32>
//       CHECK:   %[[r:.*]] = bufferization.to_tensor %[[alloc]] : memref<?x?xf32>
//       CHECK:   return %[[r]]
func.func @transfer_write(%t: tensor<?x?xf32>, %o1: index,
                     %o2: index, %vec: vector<5x6xf32>) -> tensor<?x?xf32> {
  %0 = vector.transfer_write %vec, %t[%o1, %o2] {in_bounds = [true, false]}
      : vector<5x6xf32>, tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
