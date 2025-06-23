; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx950 -verify-machineinstrs < %s | FileCheck -check-prefix=PAL %s

; GFX950supports upto 160 KB configurable LDS memory.
; This test checks the min and max size of LDS that can be allocated.

; PAL: .shader_functions:
; PAL: test_lds_array_i32:
; PAL: .lds_size:       0x28000
; PAL: test_lds_i32:
; PAL: .lds_size:       0x4


@lds.i32 = addrspace(3) global i32 poison
@lds.array.i32 = addrspace(3) global [40959 x i32] poison

define amdgpu_gfx void @test_lds_i32(i32 %val) {
  store i32 %val, ptr addrspace(3) @lds.i32
  ret void
}

define amdgpu_gfx void @test_lds_array_i32() {
  %gep = getelementptr inbounds [40959 x i32], ptr addrspace(3) @lds.array.i32, i32 0, i32 20
  %val = load i32, ptr addrspace(3) %gep
  store i32 %val, ptr addrspace(3) @lds.i32
  ret void
}