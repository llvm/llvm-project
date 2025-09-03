; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1250 < %s | FileCheck -check-prefix=PAL %s

; GFX1250 supports upto 320 KB configurable LDS memory.
; This test checks the min and max size of LDS that can be allocated.

; PAL: .shader_functions:
; PAL: test_lds_array_i16:
; PAL: .lds_size:       0x50000
; PAL: test_lds_array_i32:
; PAL: .lds_size:       0x50000
; PAL: test_lds_array_i8:
; PAL: .lds_size:       0x50000
; PAL: test_lds_i16:
; PAL: .lds_size:       0x2
; PAL: test_lds_i32:
; PAL: .lds_size:       0x4
; PAL: test_lds_i8:
; PAL: .lds_size:       0x1

@lds.i8 = addrspace(3) global i8 undef
@lds.array.i8 = addrspace(3) global [327679 x i8] undef
@lds.i16 = addrspace(3) global i16 undef
@lds.array.i16 = addrspace(3) global [163839 x i16] undef
@lds.i32 = addrspace(3) global i32 undef
@lds.array.i32 = addrspace(3) global [81919 x i32] undef

define amdgpu_gfx void @test_lds_i8(i8 %val) {
  store i8 %val, ptr addrspace(3) @lds.i8
  ret void
}

define amdgpu_gfx void @test_lds_i16(i16 %val) {
  store i16 %val, ptr addrspace(3) @lds.i16
  ret void
}

define amdgpu_gfx void @test_lds_i32(i32 %val) {
  store i32 %val, ptr addrspace(3) @lds.i32
  ret void
}

define amdgpu_gfx void @test_lds_array_i8() {
  %gep = getelementptr inbounds [327679 x i8], ptr addrspace(3) @lds.array.i8, i32 0, i32 5
  %val = load i8, ptr addrspace(3) %gep
  store i8 %val, ptr addrspace(3) @lds.i8
  ret void
}

define amdgpu_gfx void @test_lds_array_i16() {
  %gep = getelementptr inbounds [163839 x i16], ptr addrspace(3) @lds.array.i16, i32 0, i32 10
  %val = load i16, ptr addrspace(3) %gep
  store i16 %val, ptr addrspace(3) @lds.i16
  ret void
}

define amdgpu_gfx void @test_lds_array_i32() {
  %gep = getelementptr inbounds [81919 x i32], ptr addrspace(3) @lds.array.i32, i32 0, i32 20
  %val = load i32, ptr addrspace(3) %gep
  store i32 %val, ptr addrspace(3) @lds.i32
  ret void
}
