// RUN: mlir-opt %s --amdgpu-transfer-read-to-load --split-input-file | FileCheck %s

// CHECK-LABEL: func @transfer_to_maskedload_fatrawbuffer(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<4xi1>
func.func @transfer_to_maskedload_fatrawbuffer(%mem : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, %idx : index, %mask : vector<4xi1>) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0, %mask {in_bounds = [true]} : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK: %[[CST:.*]] = arith.constant 0.0
// CHECK: %[[SPLAT:.*]] = vector.splat %[[CST]]
// CHECK: %[[LOAD:.*]] = vector.load %arg0[%arg1, %arg1]
// CHECK: %[[SELECT:.*]] = arith.select %arg2, %[[LOAD]], %[[SPLAT]]
// CHECK: return %[[SELECT]] : vector<4xf32>

// -----

// CHECK-LABEL: func @transfer_to_maskedload_regular(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<4xi1>
func.func @transfer_to_maskedload_regular(%mem : memref<8x8xf32>, %idx : index, %mask : vector<4xi1>) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0, %mask {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK: %[[CST:.*]] = arith.constant 0.0
// CHECK: %[[RES:.*]] = vector.transfer_read %arg0[%arg1, %arg1], %[[CST]], %arg2 {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>
// CHECK: return %[[RES]] : vector<4xf32>

// -----

// CHECK-LABEL: func @transfer_to_maskedload_addrspace(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32, #gpu.address_space<workgroup>>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<4xi1>
func.func @transfer_to_maskedload_addrspace(%mem : memref<8x8xf32, #gpu.address_space<workgroup>>, %idx : index, %mask : vector<4xi1>) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0, %mask {in_bounds = [true]} : memref<8x8xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK: %[[CST:.*]] = arith.constant 0.0
// CHECK: %[[RES:.*]] = vector.transfer_read %arg0[%arg1, %arg1], %[[CST]], %arg2 {in_bounds = [true]} : memref<8x8xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK: return %[[RES]] : vector<4xf32>

// -----

// CHECK-LABEL: func @transfer_broadcasting(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<1xi1>
#broadcast_1d = affine_map<(d0, d1) -> (0)>
func.func @transfer_broadcasting(%mem : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, %idx : index, %mask : vector<1xi1>) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0, %mask
    {in_bounds = [true], permutation_map = #broadcast_1d}
      : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK: %[[CST:.*]] = arith.constant 0.0
// CHECK: %[[SPLAT:.*]] = vector.splat %[[CST]]
// CHECK: %[[LOAD:.*]] = vector.load %arg0[%arg1, %arg1]
// CHECK: %[[SELECT:.*]] = arith.select %arg2, %[[LOAD]], %[[SPLAT]]
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[SELECT]] : vector<1xf32> to vector<4xf32>
// CHECK: return %[[BROADCAST]] : vector<4xf32>

// -----

// CHECK-LABEL: func @transfer_scalar(
// CHECK-SAME: %[[ARG0:.*]]: memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-SAME: %[[ARG2:.*]]: vector<1xi1>
func.func @transfer_scalar(%mem : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, %idx : index, %mask : vector<1xi1>) -> vector<1xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0, %mask
    {in_bounds = [true]}
      : memref<8x8xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  return %res : vector<1xf32>
}
// CHECK: %[[CST:.*]] = arith.constant 0.0
// CHECK: %[[SPLAT:.*]] = vector.splat %[[CST]]
// CHECK: %[[LOAD:.*]] = vector.load %arg0[%arg1, %arg1]
// CHECK: %[[SELECT:.*]] = arith.select %arg2, %[[LOAD]], %[[SPLAT]]
// CHECK: return %[[SELECT]] : vector<1xf32>
