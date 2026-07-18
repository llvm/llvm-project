// RUN: mlir-opt %s --split-input-file --verify-diagnostics -convert-amdgpu-to-rocdl=chipset=gfx1250

func.func @transpose_load_to_rocdl_4xf16(%idx1 : index, %idx2 : index, %wgmem : memref<128x72xf16, 3>) -> vector<4xf16> {
  // expected-error@+2 {{'amdgpu.transpose_load' op 16-bit transpose_load requires 8 elements on gfx1250+}}
  // expected-error@+1 {{failed to legalize operation 'amdgpu.transpose_load'}}
  %0 = amdgpu.transpose_load %wgmem[%idx1, %idx2] : memref<128x72xf16, 3> -> vector<4xf16>
  return %0 : vector<4xf16>
}
