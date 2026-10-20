; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

;; i8

define void @st_global_i8(ptr addrspace(0) %ptr, i8 %a) {
; PTX32: st.b8 [%r{{[0-9]+}}], %rs{{[0-9]+}}
; PTX32: ret
; PTX64: st.b8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
; PTX64: ret
  store i8 %a, ptr addrspace(0) %ptr
  ret void
}

;; i16

define void @st_global_i16(ptr addrspace(0) %ptr, i16 %a) {
; PTX32: st.b16 [%r{{[0-9]+}}], %rs{{[0-9]+}}
; PTX32: ret
; PTX64: st.b16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
; PTX64: ret
  store i16 %a, ptr addrspace(0) %ptr
  ret void
}

;; i32

define void @st_global_i32(ptr addrspace(0) %ptr, i32 %a) {
; PTX32: st.b32 [%r{{[0-9]+}}], %r{{[0-9]+}}
; PTX32: ret
; PTX64: st.b32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
; PTX64: ret
  store i32 %a, ptr addrspace(0) %ptr
  ret void
}

;; i64

define void @st_global_i64(ptr addrspace(0) %ptr, i64 %a) {
; PTX32: st.b64 [%r{{[0-9]+}}], %rd{{[0-9]+}}
; PTX32: ret
; PTX64: st.b64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
; PTX64: ret
  store i64 %a, ptr addrspace(0) %ptr
  ret void
}

;; f32

define void @st_global_f32(ptr addrspace(0) %ptr, float %a) {
; PTX32: st.b32 [%r{{[0-9]+}}], %r{{[0-9]+}}
; PTX32: ret
; PTX64: st.b32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
; PTX64: ret
  store float %a, ptr addrspace(0) %ptr
  ret void
}

;; f64

define void @st_global_f64(ptr addrspace(0) %ptr, double %a) {
; PTX32: st.b64 [%r{{[0-9]+}}], %rd{{[0-9]+}}
; PTX32: ret
; PTX64: st.b64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
; PTX64: ret
  store double %a, ptr addrspace(0) %ptr
  ret void
}
