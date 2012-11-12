; RUN: llc < %s -march=nvptx -mcpu=sm_10 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_10 | FileCheck %s --check-prefix=PTX64
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64


;; i8

define void @st_global_i8(i8 addrspace(1)* %ptr, i8 %a) {
; PTX32: st.global.u8 [%r{{[0-9]+}}], %rc{{[0-9]+}}
; PTX32: ret
; PTX64: st.global.u8 [%rl{{[0-9]+}}], %rc{{[0-9]+}}
; PTX64: ret
  store i8 %a, i8 addrspace(1)* %ptr
  ret void
}

define void @st_shared_i8(i8 addrspace(3)* %ptr, i8 %a) {
; PTX32: st.shared.u8 [%r{{[0-9]+}}], %rc{{[0-9]+}}
; PTX32: ret
; PTX64: st.shared.u8 [%rl{{[0-9]+}}], %rc{{[0-9]+}}
; PTX64: ret
  store i8 %a, i8 addrspace(3)* %ptr
  ret void
}

define void @st_local_i8(i8 addrspace(5)* %ptr, i8 %a) {
; PTX32: st.local.u8 [%r{{[0-9]+}}], %rc{{[0-9]+}}
; PTX32: ret
; PTX64: st.local.u8 [%rl{{[0-9]+}}], %rc{{[0-9]+}}
; PTX64: ret
  store i8 %a, i8 addrspace(5)* %ptr
  ret void
}

;; i16

define void @st_global_i16(i16 addrspace(1)* %ptr, i16 %a) {
; PTX32: st.global.u16 [%r{{[0-9]+}}], %rs{{[0-9]+}}
; PTX32: ret
; PTX64: st.global.u16 [%rl{{[0-9]+}}], %rs{{[0-9]+}}
; PTX64: ret
  store i16 %a, i16 addrspace(1)* %ptr
  ret void
}

define void @st_shared_i16(i16 addrspace(3)* %ptr, i16 %a) {
; PTX32: st.shared.u16 [%r{{[0-9]+}}], %rs{{[0-9]+}}
; PTX32: ret
; PTX64: st.shared.u16 [%rl{{[0-9]+}}], %rs{{[0-9]+}}
; PTX64: ret
  store i16 %a, i16 addrspace(3)* %ptr
  ret void
}

define void @st_local_i16(i16 addrspace(5)* %ptr, i16 %a) {
; PTX32: st.local.u16 [%r{{[0-9]+}}], %rs{{[0-9]+}}
; PTX32: ret
; PTX64: st.local.u16 [%rl{{[0-9]+}}], %rs{{[0-9]+}}
; PTX64: ret
  store i16 %a, i16 addrspace(5)* %ptr
  ret void
}

;; i32

define void @st_global_i32(i32 addrspace(1)* %ptr, i32 %a) {
; PTX32: st.global.u32 [%r{{[0-9]+}}], %r{{[0-9]+}}
; PTX32: ret
; PTX64: st.global.u32 [%rl{{[0-9]+}}], %r{{[0-9]+}}
; PTX64: ret
  store i32 %a, i32 addrspace(1)* %ptr
  ret void
}

define void @st_shared_i32(i32 addrspace(3)* %ptr, i32 %a) {
; PTX32: st.shared.u32 [%r{{[0-9]+}}], %r{{[0-9]+}}
; PTX32: ret
; PTX64: st.shared.u32 [%rl{{[0-9]+}}], %r{{[0-9]+}}
; PTX64: ret
  store i32 %a, i32 addrspace(3)* %ptr
  ret void
}

define void @st_local_i32(i32 addrspace(5)* %ptr, i32 %a) {
; PTX32: st.local.u32 [%r{{[0-9]+}}], %r{{[0-9]+}}
; PTX32: ret
; PTX64: st.local.u32 [%rl{{[0-9]+}}], %r{{[0-9]+}}
; PTX64: ret
  store i32 %a, i32 addrspace(5)* %ptr
  ret void
}

;; i64

define void @st_global_i64(i64 addrspace(1)* %ptr, i64 %a) {
; PTX32: st.global.u64 [%r{{[0-9]+}}], %rl{{[0-9]+}}
; PTX32: ret
; PTX64: st.global.u64 [%rl{{[0-9]+}}], %rl{{[0-9]+}}
; PTX64: ret
  store i64 %a, i64 addrspace(1)* %ptr
  ret void
}

define void @st_shared_i64(i64 addrspace(3)* %ptr, i64 %a) {
; PTX32: st.shared.u64 [%r{{[0-9]+}}], %rl{{[0-9]+}}
; PTX32: ret
; PTX64: st.shared.u64 [%rl{{[0-9]+}}], %rl{{[0-9]+}}
; PTX64: ret
  store i64 %a, i64 addrspace(3)* %ptr
  ret void
}

define void @st_local_i64(i64 addrspace(5)* %ptr, i64 %a) {
; PTX32: st.local.u64 [%r{{[0-9]+}}], %rl{{[0-9]+}}
; PTX32: ret
; PTX64: st.local.u64 [%rl{{[0-9]+}}], %rl{{[0-9]+}}
; PTX64: ret
  store i64 %a, i64 addrspace(5)* %ptr
  ret void
}

;; f32

define void @st_global_f32(float addrspace(1)* %ptr, float %a) {
; PTX32: st.global.f32 [%r{{[0-9]+}}], %f{{[0-9]+}}
; PTX32: ret
; PTX64: st.global.f32 [%rl{{[0-9]+}}], %f{{[0-9]+}}
; PTX64: ret
  store float %a, float addrspace(1)* %ptr
  ret void
}

define void @st_shared_f32(float addrspace(3)* %ptr, float %a) {
; PTX32: st.shared.f32 [%r{{[0-9]+}}], %f{{[0-9]+}}
; PTX32: ret
; PTX64: st.shared.f32 [%rl{{[0-9]+}}], %f{{[0-9]+}}
; PTX64: ret
  store float %a, float addrspace(3)* %ptr
  ret void
}

define void @st_local_f32(float addrspace(5)* %ptr, float %a) {
; PTX32: st.local.f32 [%r{{[0-9]+}}], %f{{[0-9]+}}
; PTX32: ret
; PTX64: st.local.f32 [%rl{{[0-9]+}}], %f{{[0-9]+}}
; PTX64: ret
  store float %a, float addrspace(5)* %ptr
  ret void
}

;; f64

define void @st_global_f64(double addrspace(1)* %ptr, double %a) {
; PTX32: st.global.f64 [%r{{[0-9]+}}], %fl{{[0-9]+}}
; PTX32: ret
; PTX64: st.global.f64 [%rl{{[0-9]+}}], %fl{{[0-9]+}}
; PTX64: ret
  store double %a, double addrspace(1)* %ptr
  ret void
}

define void @st_shared_f64(double addrspace(3)* %ptr, double %a) {
; PTX32: st.shared.f64 [%r{{[0-9]+}}], %fl{{[0-9]+}}
; PTX32: ret
; PTX64: st.shared.f64 [%rl{{[0-9]+}}], %fl{{[0-9]+}}
; PTX64: ret
  store double %a, double addrspace(3)* %ptr
  ret void
}

define void @st_local_f64(double addrspace(5)* %ptr, double %a) {
; PTX32: st.local.f64 [%r{{[0-9]+}}], %fl{{[0-9]+}}
; PTX32: ret
; PTX64: st.local.f64 [%rl{{[0-9]+}}], %fl{{[0-9]+}}
; PTX64: ret
  store double %a, double addrspace(5)* %ptr
  ret void
}
