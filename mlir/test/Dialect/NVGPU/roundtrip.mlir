// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ldmatrix(
func.func @ldmatrix(%arg0: memref<?x?xf16, 3>, %x: index, %y: index) {
//      CHECK: nvgpu.ldmatrix %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK-SAME: {numTiles = 4 : i32, transpose = false} : memref<?x?xf16, 3> -> vector<4x2xf16>
  %l = nvgpu.ldmatrix %arg0[%x, %y] {numTiles = 4 : i32, transpose = false} :
    memref<?x?xf16, 3> -> vector<4x2xf16>
  return
}

// CHECK-LABEL: func @mma_sync(
func.func @mma_sync(%arg0: vector<4x2xf16>,
               %arg1: vector<2x2xf16>,
               %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
//       CHECK: nvgpu.mma.sync(%{{.*}}, %{{.*}}, %{{.*}}) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  %d = nvgpu.mma.sync(%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} :
    (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}

// CHECK-LABEL: func @mma_sp_sync_f16_16832(
func.func @mma_sp_sync_f16_16832(%arg0: vector<4x2xf16>,
                                 %arg1: vector<4x2xf16>,
                                 %arg2: vector<2x2xf16>,
                                 %arg3: vector<2xi16>) -> vector<2x2xf16> {
  //      CHECK: nvgpu.mma.sp.sync(%{{.*}}, %{{.*}}, %{{.*}}) metadata(%{{.+}}) {
  // CHECK-SAME:   mmaShape = [16, 8, 32]
  // CHECK-SAME: (vector<4x2xf16>, vector<4x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  %d = nvgpu.mma.sp.sync(%arg0, %arg1, %arg2) metadata(%arg3) {mmaShape = [16, 8, 32]} :
    (vector<4x2xf16>, vector<4x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}

// CHECK-LABEL: func @mma_sp_sync_f16_16816(
func.func @mma_sp_sync_f16_16816(%arg0: vector<2x2xf16>,
                                 %arg1: vector<2x2xf16>,
                                 %arg2: vector<2x2xf16>,
                                 %arg3: vector<2xi16>) -> vector<2x2xf16> {
  //      CHECK: nvgpu.mma.sp.sync(%{{.*}}, %{{.*}}, %{{.*}}) metadata(%{{.+}}) {
  // CHECK-SAME:   mmaShape = [16, 8, 16]
  // CHECK-SAME: (vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  %d = nvgpu.mma.sp.sync(%arg0, %arg1, %arg2) metadata(%arg3) {mmaShape = [16, 8, 16]} :
    (vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}

// CHECK-LABEL: func @mma_sp_sync_i8_16864(
func.func @mma_sp_sync_i8_16864(%arg0: vector<4x4xi8>,
                                %arg1: vector<4x4xi8>,
                                %arg2: vector<2x2xi32>,
                                %arg3: vector<2xi16>) -> vector<2x2xi32> {
  //      CHECK: nvgpu.mma.sp.sync(%{{.*}}, %{{.*}}, %{{.*}}) metadata(%{{.+}}) {
  // CHECK-SAME:   mmaShape = [16, 8, 64]
  // CHECK-SAME: (vector<4x4xi8>, vector<4x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  %d = nvgpu.mma.sp.sync(%arg0, %arg1, %arg2) metadata(%arg3) {mmaShape = [16, 8, 64]} :
    (vector<4x4xi8>, vector<4x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  return %d : vector<2x2xi32>
}

func.func @async_cp(%dst : memref<2x7x5xf32, 3>, %src : memref<4x5xf32>){
  // CHECK-LABEL: func @async_cp
  %c0 = arith.constant 0 : index
  // CHECK: nvgpu.device_async_copy %{{.*}}[{{.*}}, {{.*}}], %{{.*}}[{{.*}}, {{.*}}, {{.*}}], 4 : memref<4x5xf32> to memref<2x7x5xf32, 3>
  %0 = nvgpu.device_async_copy %src[%c0, %c0], %dst[%c0, %c0, %c0], 4 : memref<4x5xf32> to memref<2x7x5xf32, 3>
  // CHECK: %{{.*}} = nvgpu.device_async_create_group
  %token = nvgpu.device_async_create_group %0
  // CHECK: nvgpu.device_async_wait %{{.*}} {numGroups = 1 : i32}
  nvgpu.device_async_wait %token {numGroups = 1 : i32}
  return
}
