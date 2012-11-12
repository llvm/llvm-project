; RUN: llc < %s -march=nvptx -mcpu=sm_10 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_10 | FileCheck %s --check-prefix=PTX64
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64


;; i8
define i8 @ld_global_i8(i8 addrspace(1)* %ptr) {
; PTX32: ld.global.u8 %rc{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.global.u8 %rc{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i8 addrspace(1)* %ptr
  ret i8 %a
}

define i8 @ld_shared_i8(i8 addrspace(3)* %ptr) {
; PTX32: ld.shared.u8 %rc{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.shared.u8 %rc{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i8 addrspace(3)* %ptr
  ret i8 %a
}

define i8 @ld_local_i8(i8 addrspace(5)* %ptr) {
; PTX32: ld.local.u8 %rc{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.local.u8 %rc{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i8 addrspace(5)* %ptr
  ret i8 %a
}

;; i16
define i16 @ld_global_i16(i16 addrspace(1)* %ptr) {
; PTX32: ld.global.u16 %rs{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.global.u16 %rs{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i16 addrspace(1)* %ptr
  ret i16 %a
}

define i16 @ld_shared_i16(i16 addrspace(3)* %ptr) {
; PTX32: ld.shared.u16 %rs{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.shared.u16 %rs{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i16 addrspace(3)* %ptr
  ret i16 %a
}

define i16 @ld_local_i16(i16 addrspace(5)* %ptr) {
; PTX32: ld.local.u16 %rs{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.local.u16 %rs{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i16 addrspace(5)* %ptr
  ret i16 %a
}

;; i32
define i32 @ld_global_i32(i32 addrspace(1)* %ptr) {
; PTX32: ld.global.u32 %r{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.global.u32 %r{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i32 addrspace(1)* %ptr
  ret i32 %a
}

define i32 @ld_shared_i32(i32 addrspace(3)* %ptr) {
; PTX32: ld.shared.u32 %r{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.shared.u32 %r{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i32 addrspace(3)* %ptr
  ret i32 %a
}

define i32 @ld_local_i32(i32 addrspace(5)* %ptr) {
; PTX32: ld.local.u32 %r{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.local.u32 %r{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i32 addrspace(5)* %ptr
  ret i32 %a
}

;; i64
define i64 @ld_global_i64(i64 addrspace(1)* %ptr) {
; PTX32: ld.global.u64 %rl{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.global.u64 %rl{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i64 addrspace(1)* %ptr
  ret i64 %a
}

define i64 @ld_shared_i64(i64 addrspace(3)* %ptr) {
; PTX32: ld.shared.u64 %rl{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.shared.u64 %rl{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i64 addrspace(3)* %ptr
  ret i64 %a
}

define i64 @ld_local_i64(i64 addrspace(5)* %ptr) {
; PTX32: ld.local.u64 %rl{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.local.u64 %rl{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i64 addrspace(5)* %ptr
  ret i64 %a
}

;; f32
define float @ld_global_f32(float addrspace(1)* %ptr) {
; PTX32: ld.global.f32 %f{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.global.f32 %f{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load float addrspace(1)* %ptr
  ret float %a
}

define float @ld_shared_f32(float addrspace(3)* %ptr) {
; PTX32: ld.shared.f32 %f{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.shared.f32 %f{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load float addrspace(3)* %ptr
  ret float %a
}

define float @ld_local_f32(float addrspace(5)* %ptr) {
; PTX32: ld.local.f32 %f{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.local.f32 %f{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load float addrspace(5)* %ptr
  ret float %a
}

;; f64
define double @ld_global_f64(double addrspace(1)* %ptr) {
; PTX32: ld.global.f64 %fl{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.global.f64 %fl{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load double addrspace(1)* %ptr
  ret double %a
}

define double @ld_shared_f64(double addrspace(3)* %ptr) {
; PTX32: ld.shared.f64 %fl{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.shared.f64 %fl{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load double addrspace(3)* %ptr
  ret double %a
}

define double @ld_local_f64(double addrspace(5)* %ptr) {
; PTX32: ld.local.f64 %fl{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.local.f64 %fl{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load double addrspace(5)* %ptr
  ret double %a
}
