// RUN: mlir-opt %s --split-input-file -convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s
// RUN: not mlir-opt %s --split-input-file -convert-amdgpu-to-rocdl=chipset=gfx942 2>&1 | FileCheck %s --check-prefix=CHECK-OLD 

// CHECK-LABEL: func @transpose_load_to_rocdl_4xf16
func.func @transpose_load_to_rocdl_4xf16(%idx1 : index, %idx2 : index, %wgmem : memref<128x72xf16, 3>) -> vector<4xf16> {
  // CHECK: rocdl.ds.read.tr16.b64
  // CHECK-OLD: error: 'amdgpu.transpose_load' op Non-gfx950 chipset not supported
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x72xf16, 3> -> vector<4xf16>
  return %0 : vector<4xf16>
}

// -----

// CHECK-LABEL: func @transpose_load_to_rocdl_8xi8
func.func @transpose_load_to_rocdl_8xi8(%idx1 : index, %idx2 : index, %wgmem : memref<128x128xi8, 3>) -> vector<8xi8> {
  // CHECK: %[[RES:.*]] = rocdl.ds.read.tr8.b64
  // CHECK-SAME: -> vector<2xi32>
  // CHECK-NEXT: llvm.bitcast %[[RES]] : vector<2xi32> to vector<8xi8>
  // CHECK-OLD: error: 'amdgpu.transpose_load' op Non-gfx950 chipset not supported
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x128xi8, 3> -> vector<8xi8>
  return %0 : vector<8xi8>
}

// -----

// CHECK-LABEL: func @transpose_load_to_rocdl_i4_memrefxi8
func.func @transpose_load_to_rocdl_i4_memrefxi8(%idx1 : index, %idx2 : index, %wgmem : memref<128x32xi8, 3>) -> vector<16xi4> {
  // CHECK: %[[RES:.*]] = rocdl.ds.read.tr4.b64
  // CHECK-SAME: -> vector<2xi32>
  // CHECK-NEXT: llvm.bitcast %[[RES]] : vector<2xi32> to vector<16xi4>
  // CHECK-OLD: error: 'amdgpu.transpose_load' op Non-gfx950 chipset not supported
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x32xi8, 3> -> vector<16xi4>
  return %0 : vector<16xi4>
}

// -----

// CHECK-LABEL: func @transpose_load_to_rocdl_i6_memrefxi8
func.func @transpose_load_to_rocdl_i6_memrefxi8(%idx1 : index, %idx2 : index, %wgmem : memref<128x32xi8, 3>) -> vector<16xi6> {
  // CHECK: %[[RES:.*]] = rocdl.ds.read.tr6.b96
  // CHECK-SAME: -> vector<3xi32>
  // CHECK-NEXT: llvm.bitcast %[[RES]] : vector<3xi32> to vector<16xi6>
  // CHECK-OLD: error: 'amdgpu.transpose_load' op Non-gfx950 chipset not supported
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x32xi8, 3> -> vector<16xi6>
  return %0 : vector<16xi6>
}

// -----

// CHECK-LABEL: func @transpose_load_to_rocdl_i16_memrefxi8
func.func @transpose_load_to_rocdl_i16_memrefxi8(%idx1 : index, %idx2 : index, %wgmem : memref<128x32xi8, 3>) -> vector<4xi16> {
  // CHECK: rocdl.ds.read.tr16.b64
  // CHECK-OLD: error: 'amdgpu.transpose_load' op Non-gfx950 chipset not supported
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x32xi8, 3> -> vector<4xi16>
  return %0 : vector<4xi16>
}
