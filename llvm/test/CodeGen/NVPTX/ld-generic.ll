; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64


;; i8
define i8 @ld_global_i8(i8 addrspace(0)* %ptr) {
; PTX32: ld.u8 %rc{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.u8 %rc{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i8 addrspace(0)* %ptr
  ret i8 %a
}

;; i16
define i16 @ld_global_i16(i16 addrspace(0)* %ptr) {
; PTX32: ld.u16 %rs{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.u16 %rs{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i16 addrspace(0)* %ptr
  ret i16 %a
}

;; i32
define i32 @ld_global_i32(i32 addrspace(0)* %ptr) {
; PTX32: ld.u32 %r{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.u32 %r{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i32 addrspace(0)* %ptr
  ret i32 %a
}

;; i64
define i64 @ld_global_i64(i64 addrspace(0)* %ptr) {
; PTX32: ld.u64 %rl{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.u64 %rl{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load i64 addrspace(0)* %ptr
  ret i64 %a
}

;; f32
define float @ld_global_f32(float addrspace(0)* %ptr) {
; PTX32: ld.f32 %f{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.f32 %f{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load float addrspace(0)* %ptr
  ret float %a
}

;; f64
define double @ld_global_f64(double addrspace(0)* %ptr) {
; PTX32: ld.f64 %fl{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.f64 %fl{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: ret
  %a = load double addrspace(0)* %ptr
  ret double %a
}
