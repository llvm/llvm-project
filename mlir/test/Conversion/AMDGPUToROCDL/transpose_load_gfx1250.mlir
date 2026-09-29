// RUN: mlir-opt %s --split-input-file -convert-amdgpu-to-rocdl=chipset=gfx1250 | FileCheck %s

// CHECK-LABEL: func @transpose_load_to_rocdl_8xf16
func.func @transpose_load_to_rocdl_8xf16(%idx1 : index, %idx2 : index, %wgmem : memref<128x72xf16, 3>) -> vector<8xf16> {
  // CHECK: rocdl.ds.load.tr16.b128
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x72xf16, 3> -> vector<8xf16>
  return %0 : vector<8xf16>
}

// -----

// CHECK-LABEL: func @transpose_load_to_rocdl_8xi8
func.func @transpose_load_to_rocdl_8xi8(%idx1 : index, %idx2 : index, %wgmem : memref<128x128xi8, 3>) -> vector<8xi8> {
  // CHECK: %[[RES:.*]] = rocdl.ds.load.tr8.b64
  // CHECK-SAME: -> vector<2xi32>
  // CHECK-NEXT: llvm.bitcast %[[RES]] : vector<2xi32> to vector<8xi8>
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x128xi8, 3> -> vector<8xi8>
  return %0 : vector<8xi8>
}

// -----

// CHECK-LABEL: func @transpose_load_to_rocdl_16xi4
func.func @transpose_load_to_rocdl_16xi4(%idx1 : index, %idx2 : index, %wgmem : memref<128x32xi8, 3>) -> vector<16xi4> {
  // CHECK: %[[RES:.*]] = rocdl.ds.load.tr4.b64
  // CHECK-SAME: -> vector<2xi32>
  // CHECK-NEXT: llvm.bitcast %[[RES]] : vector<2xi32> to vector<16xi4>
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x32xi8, 3> -> vector<16xi4>
  return %0 : vector<16xi4>
}

// -----

// CHECK-LABEL: func @transpose_load_to_rocdl_16xi6
func.func @transpose_load_to_rocdl_16xi6(%idx1 : index, %idx2 : index, %wgmem : memref<128x32xi8, 3>) -> vector<16xi6> {
  // CHECK: %[[RES:.*]] = rocdl.ds.load.tr6.b96
  // CHECK-SAME: -> vector<3xi32>
  // CHECK-NEXT: llvm.bitcast %[[RES]] : vector<3xi32> to vector<16xi6>
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x32xi8, 3> -> vector<16xi6>
  return %0 : vector<16xi6>
}
