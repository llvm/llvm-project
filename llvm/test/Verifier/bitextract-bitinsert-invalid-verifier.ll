; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: bitinsert val type cannot be wider than base type!
; CHECK-NEXT: %r = bitinsert b32 %base, i64 %val, i32 0
define b32 @bitinsert_val_wider_than_base(b32 %base, i64 %val) {
  %r = bitinsert b32 %base, i64 %val, i32 0
  ret b32 %r
}

; CHECK: bitinsert val type cannot be wider than base type!
; CHECK-NEXT: %r = bitinsert b48 %base, ptr %val, i32 0
define b48 @bitinsert_val_ptr_wider_than_base(b48 %base, ptr %val) {
  %r = bitinsert b48 %base, ptr %val, i32 0
  ret b48 %r
}

; CHECK: bitextract result type cannot be wider than source type!
; CHECK-NEXT: %r = bitextract i64, b32 %src, i32 0
define i64 @bitextract_ty_wider_than_src(b32 %src) {
  %r = bitextract i64, b32 %src, i32 0
  ret i64 %r
}

; CHECK: bitextract result type cannot be wider than source type!
; CHECK-NEXT: %r = bitextract ptr, b48 %src, i32 0
define ptr @bitextract_ty_ptr_wider_than_src(b48 %src) {
  %r = bitextract ptr, b48 %src, i32 0
  ret ptr %r
}

; CHECK: bitinsert val type cannot be wider than base type!
; CHECK-NEXT: %r = bitinsert <2 x b32> %base, <2 x i64> %val, <2 x i32> zeroinitializer
define <2 x b32> @bitinsert_vec_val_wider_than_base(<2 x b32> %base,
                                                    <2 x i64> %val) {
  %r = bitinsert <2 x b32> %base, <2 x i64> %val, <2 x i32> zeroinitializer
  ret <2 x b32> %r
}

; CHECK: bitextract result type cannot be wider than source type!
; CHECK-NEXT: %r = bitextract <2 x i64>, <2 x b32> %src, <2 x i32> zeroinitializer
define <2 x i64> @bitextract_vec_ty_wider_than_src(<2 x b32> %src) {
  %r = bitextract <2 x i64>, <2 x b32> %src, <2 x i32> zeroinitializer
  ret <2 x i64> %r
}
