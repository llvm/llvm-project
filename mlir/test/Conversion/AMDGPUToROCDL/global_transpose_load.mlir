// RUN: mlir-opt %s --split-input-file --verify-diagnostics -convert-amdgpu-to-rocdl=chipset=gfx1201 | FileCheck %s
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

func.func @global_transpose_load_wrong_addrspace(%i : index, %j : index,
    %src : memref<128x256xf16, 3>) -> vector<8xf16> {
  // expected-error@+1 {{'amdgpu.global_transpose_load' op source memory address space must be Global}}
  %0 = amdgpu.global_transpose_load %src[%i, %j]
         : memref<128x256xf16, 3> -> vector<8xf16>
  return %0 : vector<8xf16>
}

// -----

func.func @global_transpose_load_unsupported_f32(%i : index, %j : index,
    %src : memref<128x256xf32, #gpu.address_space<global>>) -> vector<8xf32> {
  // expected-error@+1 {{'amdgpu.global_transpose_load' op unsupported element type size for global transpose load: 32 bits}}
  %0 = amdgpu.global_transpose_load %src[%i, %j]
         : memref<128x256xf32, #gpu.address_space<global>> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

func.func @global_transpose_load_wrong_num_elements(%i : index, %j : index,
    %src : memref<128x256xf16, #gpu.address_space<global>>) -> vector<4xf16> {
  // expected-error@+1 {{'amdgpu.global_transpose_load' op transferring type size mismatch: expected num of elements: 8}}
  %0 = amdgpu.global_transpose_load %src[%i, %j]
         : memref<128x256xf16, #gpu.address_space<global>> -> vector<4xf16>
  return %0 : vector<4xf16>
}
