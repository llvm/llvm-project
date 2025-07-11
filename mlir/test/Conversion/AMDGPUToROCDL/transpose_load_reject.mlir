// RUN: not mlir-opt %s --split-input-file -convert-amdgpu-to-rocdl=chipset=gfx950 2>&1 | FileCheck %s

// -----

func.func @transpose_load_to_rocdl_16xi4(%idx1 : index, %idx2 : index, %wgmem : memref<128x16xi4, 3>) -> vector<16xi4> {
  // CHECK: memref to have at least 8 bits element size, got 4
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x16xi4, 3> -> vector<16xi4>
  return %0 : vector<16xi4>
}

// -----

func.func @transpose_load_to_rocdl_16xi6(%idx1 : index, %idx2 : index, %wgmem : memref<128x32xi6, 3>) -> vector<16xi6> {
  // CHECK: memref to have at least 8 bits element size, got 6
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x32xi6, 3> -> vector<16xi6>
  return %0 : vector<16xi6>
}
