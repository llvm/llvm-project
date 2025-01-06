; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}


;; i8
define i8 @ld_global_i8(ptr addrspace(0) %ptr) {
; PTX32: ld.u8 %r{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.u8 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
; PTX64: ret
  %a = load i8, ptr addrspace(0) %ptr
  ret i8 %a
}

;; i16
define i16 @ld_global_i16(ptr addrspace(0) %ptr) {
; PTX32: ld.u16 %r{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.u16 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
; PTX64: ret
  %a = load i16, ptr addrspace(0) %ptr
  ret i16 %a
}

;; i32
define i32 @ld_global_i32(ptr addrspace(0) %ptr) {
; PTX32: ld.u32 %r{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
; PTX64: ret
  %a = load i32, ptr addrspace(0) %ptr
  ret i32 %a
}

;; i64
define i64 @ld_global_i64(ptr addrspace(0) %ptr) {
; PTX32: ld.u64 %rd{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
; PTX64: ret
  %a = load i64, ptr addrspace(0) %ptr
  ret i64 %a
}

;; f32
define float @ld_global_f32(ptr addrspace(0) %ptr) {
; PTX32: ld.f32 %f{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.f32 %f{{[0-9]+}}, [%rd{{[0-9]+}}]
; PTX64: ret
  %a = load float, ptr addrspace(0) %ptr
  ret float %a
}

;; f64
define double @ld_global_f64(ptr addrspace(0) %ptr) {
; PTX32: ld.f64 %fd{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: ret
; PTX64: ld.f64 %fd{{[0-9]+}}, [%rd{{[0-9]+}}]
; PTX64: ret
  %a = load double, ptr addrspace(0) %ptr
  ret double %a
}
