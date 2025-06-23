// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s

#gpu_lds_addrspace = 3
#amdgpu_fat_buffer_addrspace = 7

// CHECK-LABEL: func @transpose_load_to_rocdl_4xf16
func.func @transpose_load_to_rocdl_4xf16(%idx1 : index, %idx2 : index, %wgmem : memref<128x72xf16, #gpu_lds_addrspace>) -> vector<4xf16> {
  // CHECK: rocdl.ds.read.tr16.b64
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x72xf16, #gpu_lds_addrspace> -> vector<4xf16>
  return %0 : vector<4xf16>
}

// CHECK-LABEL: func @transpose_load_to_rocdl_8xi8
func.func @transpose_load_to_rocdl_8xi8(%idx1 : index, %idx2 : index, %wgmem : memref<128x128xi8, #gpu_lds_addrspace>) -> vector<8xi8> {
  // CHECK: rocdl.ds.read.tr8.b64
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x128xi8, #gpu_lds_addrspace> -> vector<8xi8>
  return %0 : vector<8xi8>
}

// CHECK-LABEL: func @transpose_load_to_rocdl_16xi4
func.func @transpose_load_to_rocdl_16xi4(%idx1 : index, %idx2 : index, %wgmem : memref<128x16xi4, #gpu_lds_addrspace>) -> vector<16xi4> {
  // CHECK: rocdl.ds.read.tr4.b64
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x16xi4, #gpu_lds_addrspace> -> vector<16xi4>
  return %0 : vector<16xi4>
}

// CHECK-LABEL: func @transpose_load_to_rocdl_3xi32
func.func @transpose_load_to_rocdl_3xi32(%idx1 : index, %idx2 : index, %wgmem : memref<128x32xi32, #gpu_lds_addrspace>) -> vector<3xi32> {
  // CHECK: rocdl.ds.read.tr6.b96
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x32xi32, #gpu_lds_addrspace> -> vector<3xi32>
  return %0 : vector<3xi32>
}
