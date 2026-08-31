; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -amdgpu-enable-lower-module-lds=0 -filetype=obj < %s | llvm-readobj -r - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -amdgpu-enable-lower-module-lds=0 -filetype=asm < %s | FileCheck %s --check-prefix=ASM

; Test ABI register-size type ID generation for various function signatures.
; The type ID encodes each parameter/return by bit width: v=void, i=<=32-bit,
; l=33-64-bit. Types with the same register footprint share an encoding
; (e.g. float(float) and i32(i32) both produce "ii").

define void @void_void() {
  ret void
}

define i32 @i32_i32(i32 %x) {
  ret i32 %x
}

define void @void_ptr_i32(ptr %p, i32 %x) {
  ret void
}

define i64 @i64_i64_i64(i64 %a, i64 %b) {
  ret i64 %a
}

define float @float_float(float %x) {
  ret float %x
}

; Take the address of each function so they appear as resource nodes.
define void @taker() {
  %p0 = alloca ptr, addrspace(5)
  store volatile ptr @void_void, ptr addrspace(5) %p0
  store volatile ptr @i32_i32, ptr addrspace(5) %p0
  store volatile ptr @void_ptr_i32, ptr addrspace(5) %p0
  store volatile ptr @i64_i64_i64, ptr addrspace(5) %p0
  store volatile ptr @float_float, ptr addrspace(5) %p0
  ret void
}

define amdgpu_kernel void @kern() {
  call void @taker()
  ret void
}

; CHECK-DAG: R_AMDGPU_ABS64 void_void
; CHECK-DAG: R_AMDGPU_ABS64 i32_i32
; CHECK-DAG: R_AMDGPU_ABS64 void_ptr_i32
; CHECK-DAG: R_AMDGPU_ABS64 i64_i64_i64
; CHECK-DAG: R_AMDGPU_ABS64 float_float
; CHECK-DAG: R_AMDGPU_ABS64 taker
; CHECK-DAG: R_AMDGPU_ABS64 kern

; ASM-DAG:    .amdgpu_info void_void
; ASM-DAG:    .amdgpu_flags 0
; ASM-DAG:    .amdgpu_typeid "v"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info i32_i32
; ASM-DAG:    .amdgpu_flags 0
; ASM-DAG:    .amdgpu_typeid "ii"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info void_ptr_i32
; ASM-DAG:    .amdgpu_flags 0
; ASM-DAG:    .amdgpu_typeid "vli"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info i64_i64_i64
; ASM-DAG:    .amdgpu_flags 0
; ASM-DAG:    .amdgpu_typeid "lll"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info float_float
; ASM-DAG:    .amdgpu_flags 0
; ASM-DAG:    .amdgpu_typeid "ii"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info taker
; ASM-DAG:    .amdgpu_flags 0
; ASM-DAG:    .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info kern
; ASM-DAG:    .amdgpu_flags 2
; ASM-DAG:    .amdgpu_call taker
; ASM-DAG:    .end_amdgpu_info
