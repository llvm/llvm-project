; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "p:64:64:64:64"

; CHECK-DAG: bitinsert b64
define b64 @bitinsert_val_int(b64 %base, i32 %val) {
  %r = bitinsert b64 %base, i32 %val, i32 0
  ret b64 %r
}

; CHECK-DAG: bitinsert b64
define b64 @bitinsert_val_fp(b64 %base, double %val) {
  %r = bitinsert b64 %base, double %val, i32 0
  ret b64 %r
}

; CHECK-DAG: bitinsert b64
define b64 @bitinsert_val_byte(b64 %base, b32 %val) {
  %r = bitinsert b64 %base, b32 %val, i32 0
  ret b64 %r
}

; CHECK-DAG: bitinsert b64
define b64 @bitinsert_val_ptr(b64 %base, ptr %val) {
  %r = bitinsert b64 %base, ptr %val, i32 0
  ret b64 %r
}

; CHECK-DAG: bitinsert b64
define b64 @bitinsert_val_ptr_other_as(b64 %base, ptr addrspace(2) %val) {
  %r = bitinsert b64 %base, ptr addrspace(2) %val, i32 0
  ret b64 %r
}

; CHECK-DAG: bitextract i32
define i32 @bitextract_ty_int(b64 %src) {
  %r = bitextract i32, b64 %src, i32 0
  ret i32 %r
}

; CHECK-DAG: bitextract double
define double @bitextract_ty_fp(b64 %src) {
  %r = bitextract double, b64 %src, i32 0
  ret double %r
}

; CHECK-DAG: bitextract b32
define b32 @bitextract_ty_byte(b64 %src) {
  %r = bitextract b32, b64 %src, i32 0
  ret b32 %r
}

; CHECK-DAG: bitextract ptr
define ptr @bitextract_ty_ptr(b64 %src) {
  %r = bitextract ptr, b64 %src, i32 0
  ret ptr %r
}

; CHECK-DAG: bitextract ptr addrspace(2)
define ptr addrspace(2) @bitextract_ty_ptr_other_as(b64 %src) {
  %r = bitextract ptr addrspace(2), b64 %src, i32 0
  ret ptr addrspace(2) %r
}