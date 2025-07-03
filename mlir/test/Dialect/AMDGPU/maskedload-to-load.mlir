// RUN: mlir-opt %s --amdgpu-maskedload-to-load --split-input-file | FileCheck %s

// CHECK-LABEL: func @transfer_to_maskedload_fatrawbuffer(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<4xi1>
// CHECK-SAME: %[[ARG3:.*]]: vector<4xf32>
func.func @transfer_to_maskedload_fatrawbuffer(%mem : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, %idx : index, %mask : vector<4xi1>, %passthru : vector<4xf32>) -> vector<4xf32> {
  %res = vector.maskedload %mem[%idx, %idx], %mask, %passthru : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK: %[[IF:.*]] = scf.if
// CHECK: vector.maskedload %[[ARG0]][%[[ARG1]], %[[ARG1]]]

// CHECK: } else {
// CHECK: %[[LOAD:.*]] = vector.load %arg0[%arg1, %arg1]
// CHECK: %[[SELECT:.*]] = arith.select %[[ARG2]], %[[LOAD]]

// CHECK: return %[[IF]] : vector<4xf32>

// -----

// CHECK: #map = affine_map<()[s0, s1] -> (s0 * 8 + s1)>
// CHECK-LABEL: func @transfer_to_maskedload_fatrawbuffer_f16(
// CHECK-SAME: %[[ARG0:.+]]: memref<8x8xf16, #amdgpu.address_space<fat_raw_buffer>>,
// CHECK-SAME: %[[ARG1:.+]]: index, %[[ARG2:.+]]: index,
// CHECK-SAME: %[[ARG3:.+]]: vector<4xi1>
// CHECK-SAME: %[[ARG4:.+]]: vector<4xf16>
func.func @transfer_to_maskedload_fatrawbuffer_f16(%mem : memref<8x8xf16, #amdgpu.address_space<fat_raw_buffer>>, %idx0 : index, %idx1 : index, %mask : vector<4xi1>, %passthru : vector<4xf16>) -> vector<4xf16> {
  %res = vector.maskedload %mem[%idx0, %idx1], %mask, %passthru : memref<8x8xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
  return %res : vector<4xf16>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[SIZE:.*]] = arith.constant 64
// CHECK-DAG: %[[BYTES:.*]] = arith.constant 2
// CHECK-DAG: %[[C4:.*]] = arith.constant 4

// CHECK: %[[LINEAR:.*]] = affine.apply #map()[%[[ARG1]], %[[ARG2]]]
// CHECK: %[[DELTA:.*]] = arith.subi %[[SIZE]], %[[LINEAR]]
// CHECK: %[[COND1:.*]] = arith.cmpi ult, %[[DELTA]], %[[C4]]

// CHECK: %[[REM:.*]] = arith.remui %[[DELTA]], %[[BYTES]]
// CHECK: %[[COND2:.*]] = arith.cmpi ne, %[[REM]], %[[C0]]

// CHECK: %[[COND:.*]] = arith.andi %[[COND1]], %[[COND2]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]] -> (vector<4xf16>) {
// CHECK: vector.maskedload %[[ARG0]][%[[ARG1]], %[[ARG2]]]
// CHECK: } else {
// CHECK: %[[LOAD:.*]] = vector.load %[[ARG0]][%[[ARG1]], %[[ARG2]]]
// CHECK: return %[[IF]] : vector<4xf16>

// -----

// CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 * s1, s2)>
// CHECK: func @transfer_to_maskedload_fatrawbuffer_dynamic_i8(
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xi8, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG1:.*]]: index, %[[ARG2:.*]]: index
// CHECK-SAME: %[[ARG3:.*]]: vector<4xi1>
// CHECK-SAME: %[[ARG4:.*]]: vector<4xi8>
func.func @transfer_to_maskedload_fatrawbuffer_dynamic_i8(%mem : memref<?x?xi8, #amdgpu.address_space<fat_raw_buffer>>, %idx0 : index, %idx1 : index, %mask : vector<4xi1>, %passthru : vector<4xi8>) -> vector<4xi8> {
  %res = vector.maskedload %mem[%idx0, %idx1], %mask, %passthru : memref<?x?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi1>, vector<4xi8> into vector<4xi8>
  return %res : vector<4xi8>
}
// CHECK:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:     %[[C4:.*]] = arith.constant 4 : index
// CHECK:     %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG0]]
// CHECK-DAG: %[[SIZE:.*]] = affine.max #[[MAP1]]()[%[[STRIDES]]#0, %[[SIZES]]#0, %[[SIZES]]#1]
// CHECK-DAG: %[[LINEAR:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[STRIDES]]#0, %[[ARG2]]]
// CHECK:     %[[IF:.*]] = scf.if
// CHECK:     return

// -----

// CHECK-LABEL: func @transfer_to_maskedload_regular(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<4xi1>
// CHECK-SAME: %[[ARG3:.*]]: vector<4xf32>
func.func @transfer_to_maskedload_regular(%mem : memref<8x8xf32>, %idx : index, %mask : vector<4xi1>, %passthru : vector<4xf32>) -> vector<4xf32> {
  %res = vector.maskedload %mem[%idx, %idx], %mask, %passthru : memref<8x8xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK: %[[RES:.*]] = vector.maskedload %[[ARG0]][%[[ARG1]], %[[ARG1]]], %[[ARG2]], %[[ARG3]]
// CHECK: return %[[RES]] : vector<4xf32>

// -----

// CHECK-LABEL: func @transfer_to_maskedload_addrspace(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32, #gpu.address_space<workgroup>>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<4xi1>
// CHECK-SAME: %[[ARG3:.*]]: vector<4xf32>
func.func @transfer_to_maskedload_addrspace(%mem : memref<8x8xf32, #gpu.address_space<workgroup>>, %idx : index, %mask : vector<4xi1>, %passthru : vector<4xf32>) -> vector<4xf32> {
  %res = vector.maskedload %mem[%idx, %idx], %mask, %passthru : memref<8x8xf32, #gpu.address_space<workgroup>>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK: %[[RES:.*]] = vector.maskedload %[[ARG0]][%[[ARG1]], %[[ARG1]]], %[[ARG2]], %[[ARG3]]
// CHECK: return %[[RES]] : vector<4xf32>

// -----

// CHECK-LABEL: func @transfer_scalar(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<1xi1>
// CHECK-SAME: %[[ARG3:.*]]: vector<1xf32>
func.func @transfer_scalar(%mem : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, %idx : index, %mask : vector<1xi1>, %passthru : vector<1xf32>) -> vector<1xf32> {
  %res = vector.maskedload %mem[%idx, %idx], %mask, %passthru
      : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi1>, vector<1xf32> into vector<1xf32>
  return %res : vector<1xf32>
}
// CHECK: %[[IF:.*]] = scf.if
// CHECK: %[[LOAD:.*]] = vector.load %[[ARG0]][%[[ARG1]], %[[ARG1]]]
// CHECK: %[[SELECT:.*]] = arith.select %arg2, %[[LOAD]], %[[ARG3]]
