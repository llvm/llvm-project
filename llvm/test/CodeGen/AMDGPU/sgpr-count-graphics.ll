; RUN: llc -global-isel -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 < %s | FileCheck %s
; RUN: llc -global-isel -mtriple=amdgcn-amd-amdpal -mcpu=tahiti < %s | FileCheck %s

@global = addrspace(1) global i32 poison, align 4

; The hardware initializes the registers received as arguments by entry points,
; so they will be counted even if unused.

; Vectors of i1 are always unpacked

; CHECK-LABEL: vec_of_i1:
; CHECK:  TotalNumSgprs: 8
define amdgpu_ps void @vec_of_i1(<8 x i1> inreg %v8i1) {
  ret void
}

; Vectors of i8 are always unpacked

; CHECK-LABEL: vec_of_i8:
; CHECK:  TotalNumSgprs: 4
define amdgpu_ps void @vec_of_i8(<4 x i8> inreg %v4i8) {
  ret void
}

; Vectors of 16-bit types are packed.
; CHECK-LABEL: vec_of_16_bit_ty:
; CHECK: TotalNumSgprs: 3
define amdgpu_ps void @vec_of_16_bit_ty(<2 x i16> inreg %v2i16, <4 x half> inreg %v4half) {
  ret void
}

; CHECK-LABEL: buffer_fat_ptr:
; CHECK: TotalNumSgprs: 5
define amdgpu_ps void @buffer_fat_ptr(ptr addrspace(7) inreg %p) {
  ret void
}
