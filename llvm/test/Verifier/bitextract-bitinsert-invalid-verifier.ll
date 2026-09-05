; RUN: not llvm-as %s 2>&1 | FileCheck %s

target datalayout = "ni:1-p:64:64:64:64"

; CHECK-DAG: bitinsert val type cannot be wider than base type!
define b32 @bitinsert_val_wider_than_base(b32 %base, i64 %val) {
  %r = bitinsert b32 %base, i64 %val, i32 0
  ret b32 %r
}

; CHECK-DAG: bitinsert val type cannot be wider than base type!
define b48 @bitinsert_val_ptr_wider_than_base(b48 %base, ptr %val) {
  %r = bitinsert b48 %base, ptr %val, i32 0
  ret b48 %r
}

; CHECK-DAG: bitinsert not supported for non-integral pointer types
define b64 @bitinsert_val_non_integral(b64 %base, ptr addrspace(1) %val) {
  %r = bitinsert b64 %base, ptr addrspace(1) %val, i32 0
  ret b64 %r
}

; CHECK-DAG: bitextract result type cannot be wider than source type!
define i64 @bitextract_ty_wider_than_src(b32 %src) {
  %r = bitextract i64, b32 %src, i32 0
  ret i64 %r
}

; CHECK-DAG: bitextract result type cannot be wider than source type!
define ptr @bitextract_ty_ptr_wider_than_src(b48 %src) {
  %r = bitextract ptr, b48 %src, i32 0
  ret ptr %r
}

; CHECK-DAG: bitextract not supported for non-integral pointer types
define ptr addrspace(1) @bitextract_ty_non_integral(b64 %src) {
  %r = bitextract ptr addrspace(1), b64 %src, i32 0
  ret ptr addrspace(1) %r
}