// RUN: mlir-opt %s --split-input-file --verify-diagnostics -convert-amdgpu-to-rocdl=chipset=gfx950

func.func @transpose_load_to_rocdl_8xf16(%idx1 : index, %idx2 : index, %wgmem : memref<128x72xf16, 3>) -> vector<8xf16> {
  // expected-error@+2 {{'amdgpu.transpose_load' op 16-bit transpose_load requires 4 elements on gfx950}}
  // expected-error@+1 {{failed to legalize operation 'amdgpu.transpose_load'}}
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x72xf16, 3> -> vector<8xf16>
  return %0 : vector<8xf16>
}
