// RUN: mlir-opt %s --split-input-file --verify-diagnostics -convert-amdgpu-to-rocdl=chipset=gfx1201 | FileCheck %s
// RUN: mlir-opt %s --split-input-file -convert-amdgpu-to-rocdl=chipset=gfx1250 | FileCheck %s --check-prefixes=CHECK,CHECK-GFX1250
// RUN: not mlir-opt %s --split-input-file -convert-amdgpu-to-rocdl=chipset=gfx942 2>&1 | FileCheck %s --check-prefix=CHECK-OLD

// CHECK-LABEL: func @global_transpose_load_8xf16
func.func @global_transpose_load_8xf16(%i : index, %j : index,
    %src : memref<128x256xf16, #gpu.address_space<global>>) -> vector<8xf16> {
  // CHECK: rocdl.global.load.tr.b128
  // CHECK-OLD: error: 'amdgpu.global_transpose_load' op global_transpose_load is only supported on gfx1200+
  %0 = amdgpu.global_transpose_load %src[%i, %j]
         : memref<128x256xf16, #gpu.address_space<global>> -> vector<8xf16>
  return %0 : vector<8xf16>
}

// -----

// CHECK-LABEL: func @global_transpose_load_8xi8
func.func @global_transpose_load_8xi8(%i : index, %j : index,
    %src : memref<128x256xi8, #gpu.address_space<global>>) -> vector<8xi8> {
  // CHECK: %[[RES:.*]] = rocdl.global.load.tr.b64
  // CHECK-SAME: -> vector<2xi32>
  // CHECK-NEXT: llvm.bitcast %[[RES]] : vector<2xi32> to vector<8xi8>
  // CHECK-OLD: error: 'amdgpu.global_transpose_load' op global_transpose_load is only supported on gfx1200+
  %0 = amdgpu.global_transpose_load %src[%i, %j]
         : memref<128x256xi8, #gpu.address_space<global>> -> vector<8xi8>
  return %0 : vector<8xi8>
}

// -----

// CHECK-GFX1250-LABEL: func @global_transpose_load_16xi4
func.func @global_transpose_load_16xi4(%i : index, %j : index,
    %src : memref<128x32xi8, #gpu.address_space<global>>) -> vector<16xi4> {
  // CHECK-GFX1250: %[[RES:.*]] = rocdl.global.load.tr4.b64
  // CHECK-GFX1250-SAME: -> vector<2xi32>
  // CHECK-GFX1250-NEXT: llvm.bitcast %[[RES]] : vector<2xi32> to vector<16xi4>
  // expected-error@+2 {{'amdgpu.global_transpose_load' op 4-bit global_transpose_load requires gfx1250+}}
  // expected-error@+1 {{failed to legalize operation 'amdgpu.global_transpose_load'}}
  %0 = amdgpu.global_transpose_load %src[%i, %j]
         : memref<128x32xi8, #gpu.address_space<global>> -> vector<16xi4>
  return %0 : vector<16xi4>
}

// -----

// CHECK-GFX1250-LABEL: func @global_transpose_load_16xi6
func.func @global_transpose_load_16xi6(%i : index, %j : index,
    %src : memref<128x32xi8, #gpu.address_space<global>>) -> vector<16xi6> {
  // CHECK-GFX1250: %[[RES:.*]] = rocdl.global.load.tr6.b96
  // CHECK-GFX1250-SAME: -> vector<3xi32>
  // CHECK-GFX1250-NEXT: llvm.bitcast %[[RES]] : vector<3xi32> to vector<16xi6>
  // expected-error@+2 {{'amdgpu.global_transpose_load' op 6-bit global_transpose_load requires gfx1250+}}
  // expected-error@+1 {{failed to legalize operation 'amdgpu.global_transpose_load'}}
  %0 = amdgpu.global_transpose_load %src[%i, %j]
         : memref<128x32xi8, #gpu.address_space<global>> -> vector<16xi6>
  return %0 : vector<16xi6>
}
