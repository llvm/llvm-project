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
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[o1:.*]]: index, %[[o2:.*]]: index, %[[vec:.*]]: vector<5x6xf32>, %[[mask:.*]]: vector<5x6xi1>)
//       CHECK:   %[[m:.*]] = bufferization.to_memref %[[t]] : memref<?x?xf32>
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}}, %{{.*}}) {{.*}} : memref<?x?xf32>
//       CHECK:   memref.copy %[[m]], %[[alloc]]
//       CHECK:   vector.transfer_write %[[vec]], %[[alloc]][%[[o1]], %[[o2]]], %[[mask]] {in_bounds = [true, false]} : vector<5x6xf32>, memref<?x?xf32>
//       CHECK:   %[[r:.*]] = bufferization.to_tensor %[[alloc]] : memref<?x?xf32>
//       CHECK:   return %[[r]]
func.func @transfer_write(%t: tensor<?x?xf32>, %o1: index,
                          %o2: index, %vec: vector<5x6xf32>,
                          %mask: vector<5x6xi1>) -> tensor<?x?xf32> {
  %0 = vector.transfer_write %vec, %t[%o1, %o2], %mask {in_bounds = [true, false]}
      : vector<5x6xf32>, tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @gather(
//  CHECK-SAME:     %[[base:.*]]: tensor<?x?xf32>, %[[v:.*]]: vector<16xi32>,
//  CHECK-SAME:     %[[mask:.*]]: vector<16xi1>, %[[pass_thru:.*]]: vector<16xf32>)
//       CHECK:   %[[m:.*]] = bufferization.to_memref %[[base]] : memref<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[out:.*]] = vector.gather %[[m]][%[[c0]], %[[c0]]] [%[[v]]], %[[mask]], %[[pass_thru]] : memref<?x?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
func.func @gather(%base: tensor<?x?xf32>, %v: vector<16xi32>, %mask: vector<16xi1>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0, %c0][%v], %mask, %pass_thru : tensor<?x?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %0 : vector<16xf32>
}

// TODO: Add test case for vector.mask. The masked op can currently not
// bufferize out-of-place, so the only test case is in one-shot-bufferize.mlir.
