; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefixes=ALL,G32,LS32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefixes=ALL,G64,LS64
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 --nvptx-short-ptr | FileCheck %s --check-prefixes=G64,LS32
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 --nvptx-short-ptr | %ptxas-verify %}


;; i8
define i8 @ld_global_i8(ptr addrspace(1) %ptr) {
; ALL-LABEL: ld_global_i8
; G32: ld.global.b8 %{{.*}}, [%r{{[0-9]+}}]
; G64: ld.global.b8 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i8, ptr addrspace(1) %ptr
  ret i8 %a
}
define i8 @ld_shared_i8(ptr addrspace(3) %ptr) {
; ALL-LABEL: ld_shared_i8
; LS32: ld.shared.b8 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.shared.b8 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i8, ptr addrspace(3) %ptr
  ret i8 %a
}
define i8 @ld_local_i8(ptr addrspace(5) %ptr) {
; ALL-LABEL: ld_local_i8
; LS32: ld.local.b8 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.local.b8 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i8, ptr addrspace(5) %ptr
  ret i8 %a
}

;; i16
define i16 @ld_global_i16(ptr addrspace(1) %ptr) {
; ALL-LABEL: ld_global_i16
; G32: ld.global.b16 %{{.*}}, [%r{{[0-9]+}}]
; G64: ld.global.b16 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i16, ptr addrspace(1) %ptr
  ret i16 %a
}
define i16 @ld_shared_i16(ptr addrspace(3) %ptr) {
; ALL-LABEL: ld_shared_i16
; LS32: ld.shared.b16 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.shared.b16 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i16, ptr addrspace(3) %ptr
  ret i16 %a
}
define i16 @ld_local_i16(ptr addrspace(5) %ptr) {
; ALL-LABEL: ld_local_i16
; LS32: ld.local.b16 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.local.b16 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i16, ptr addrspace(5) %ptr
  ret i16 %a
}

;; i32
define i32 @ld_global_i32(ptr addrspace(1) %ptr) {
; ALL-LABEL: ld_global_i32
; G32: ld.global.b32 %{{.*}}, [%r{{[0-9]+}}]
; G64: ld.global.b32 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i32, ptr addrspace(1) %ptr
  ret i32 %a
}
define i32 @ld_shared_i32(ptr addrspace(3) %ptr) {
; ALL-LABEL: ld_shared_i32
; LS32: ld.shared.b32 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.shared.b32 %{{.*}}, [%rd{{[0-9]+}}]
; PTX64: ret
  %a = load i32, ptr addrspace(3) %ptr
  ret i32 %a
}
define i32 @ld_local_i32(ptr addrspace(5) %ptr) {
; ALL-LABEL: ld_local_i32
; LS32: ld.local.b32 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.local.b32 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i32, ptr addrspace(5) %ptr
  ret i32 %a
}

;; i64
define i64 @ld_global_i64(ptr addrspace(1) %ptr) {
; ALL-LABEL: ld_global_i64
; G32: ld.global.b64 %{{.*}}, [%r{{[0-9]+}}]
; G64: ld.global.b64 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i64, ptr addrspace(1) %ptr
  ret i64 %a
}
define i64 @ld_shared_i64(ptr addrspace(3) %ptr) {
; ALL-LABEL: ld_shared_i64
; LS32: ld.shared.b64 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.shared.b64 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i64, ptr addrspace(3) %ptr
  ret i64 %a
}
define i64 @ld_local_i64(ptr addrspace(5) %ptr) {
; ALL-LABEL: ld_local_i64
; LS32: ld.local.b64 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.local.b64 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load i64, ptr addrspace(5) %ptr
  ret i64 %a
}

;; f32
define float @ld_global_f32(ptr addrspace(1) %ptr) {
; ALL-LABEL: ld_global_f32
; G32: ld.global.b32 %{{.*}}, [%r{{[0-9]+}}]
; G64: ld.global.b32 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load float, ptr addrspace(1) %ptr
  ret float %a
}
define float @ld_shared_f32(ptr addrspace(3) %ptr) {
; ALL-LABEL: ld_shared_f32
; LS32: ld.shared.b32 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.shared.b32 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load float, ptr addrspace(3) %ptr
  ret float %a
}
define float @ld_local_f32(ptr addrspace(5) %ptr) {
; ALL-LABEL: ld_local_f32
; LS32: ld.local.b32 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.local.b32 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load float, ptr addrspace(5) %ptr
  ret float %a
}

;; f64
define double @ld_global_f64(ptr addrspace(1) %ptr) {
; ALL-LABEL: ld_global_f64
; G32: ld.global.b64 %{{.*}}, [%r{{[0-9]+}}]
; G64: ld.global.b64 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load double, ptr addrspace(1) %ptr
  ret double %a
}
define double @ld_shared_f64(ptr addrspace(3) %ptr) {
; ALL-LABEL: ld_shared_f64
; LS32: ld.shared.b64 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.shared.b64 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load double, ptr addrspace(3) %ptr
  ret double %a
}
define double @ld_local_f64(ptr addrspace(5) %ptr) {
; ALL-LABEL: ld_local_f64
; LS32: ld.local.b64 %{{.*}}, [%r{{[0-9]+}}]
; LS64: ld.local.b64 %{{.*}}, [%rd{{[0-9]+}}]
; ALL: ret
  %a = load double, ptr addrspace(5) %ptr
  ret double %a
}
