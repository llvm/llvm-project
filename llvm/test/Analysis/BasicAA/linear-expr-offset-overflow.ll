; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; For x == -1317624576693539401, the index is 1. Each operation is
; defined, but folding INT64_MAX + 1 overflows the signed index type.
; CHECK-LABEL: Function: add_offset_overflow
; CHECK: MayAlias: i8* %g, i16* %p
define void @add_offset_overflow(ptr %p, i64 %x) {
  %m = mul nsw i64 %x, 7
  %a = add nsw i64 %m, 9223372036854775807
  %b = add nsw i64 %a, 1
  %g = getelementptr i8, ptr %p, i64 %b
  load i8, ptr %g
  load i16, ptr %p
  ret void
}

; The same index computed by subtracting -1.
; CHECK-LABEL: Function: sub_offset_overflow
; CHECK: MayAlias: i8* %g, i16* %p
define void @sub_offset_overflow(ptr %p, i64 %x) {
  %m = mul nsw i64 %x, 7
  %a = add nsw i64 %m, 9223372036854775807
  %b = sub nsw i64 %a, -1
  %g = getelementptr i8, ptr %p, i64 %b
  load i8, ptr %g
  load i16, ptr %p
  ret void
}

; Folding the constants to 2 does not overflow, so retain nsw for 7*x.
; CHECK-LABEL: Function: add_offset_no_overflow
; CHECK: NoAlias: i8* %g, i8* %p
define void @add_offset_no_overflow(ptr %p, i64 %x) {
  %m = mul nsw i64 %x, 7
  %a = add nsw i64 %m, 1
  %b = add nsw i64 %a, 1
  %g = getelementptr i8, ptr %p, i64 %b
  load i8, ptr %g
  load i8, ptr %p
  ret void
}

; Folding the constants to 2 does not overflow, so retain nsw for 7*x.
; CHECK-LABEL: Function: sub_offset_no_overflow
; CHECK: NoAlias: i8* %g, i8* %p
define void @sub_offset_no_overflow(ptr %p, i64 %x) {
  %m = mul nsw i64 %x, 7
  %a = add nsw i64 %m, 3
  %b = sub nsw i64 %a, 1
  %g = getelementptr i8, ptr %p, i64 %b
  load i8, ptr %g
  load i8, ptr %p
  ret void
}
