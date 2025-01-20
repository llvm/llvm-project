; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

;; i8

define void @st_global_i8(ptr addrspace(0) %ptr, i8 %a) {
; PTX32: st.u8 [%r{{[0-9]+}}], %rs{{[0-9]+}}
; PTX32: ret
; PTX64: st.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
; PTX64: ret
  store i8 %a, ptr addrspace(0) %ptr
  ret void
}

;; i16

define void @st_global_i16(ptr addrspace(0) %ptr, i16 %a) {
; PTX32: st.u16 [%r{{[0-9]+}}], %rs{{[0-9]+}}
; PTX32: ret
; PTX64: st.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
; PTX64: ret
  store i16 %a, ptr addrspace(0) %ptr
  ret void
}

;; i32

define void @st_global_i32(ptr addrspace(0) %ptr, i32 %a) {
; PTX32: st.u32 [%r{{[0-9]+}}], %r{{[0-9]+}}
; PTX32: ret
; PTX64: st.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
; PTX64: ret
  store i32 %a, ptr addrspace(0) %ptr
  ret void
}

;; i64

define void @st_global_i64(ptr addrspace(0) %ptr, i64 %a) {
; PTX32: st.u64 [%r{{[0-9]+}}], %rd{{[0-9]+}}
; PTX32: ret
; PTX64: st.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
; PTX64: ret
  store i64 %a, ptr addrspace(0) %ptr
  ret void
}

;; f32

define void @st_global_f32(ptr addrspace(0) %ptr, float %a) {
; PTX32: st.f32 [%r{{[0-9]+}}], %f{{[0-9]+}}
; PTX32: ret
; PTX64: st.f32 [%rd{{[0-9]+}}], %f{{[0-9]+}}
; PTX64: ret
  store float %a, ptr addrspace(0) %ptr
  ret void
}

;; f64

define void @st_global_f64(ptr addrspace(0) %ptr, double %a) {
; PTX32: st.f64 [%r{{[0-9]+}}], %fd{{[0-9]+}}
; PTX32: ret
; PTX64: st.f64 [%rd{{[0-9]+}}], %fd{{[0-9]+}}
; PTX64: ret
  store double %a, ptr addrspace(0) %ptr
  ret void
}
