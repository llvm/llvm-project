; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"

; CHECK-LABEL: Function: trunc_offset
; CHECK: NoAlias: i32* %p, i32* %q
define void @trunc_offset(ptr %base, i64 %x) {
  %next = add i64 %x, 1
  %i = trunc i64 %x to i32
  %j = trunc i64 %next to i32
  %p = getelementptr i32, ptr %base, i32 %i
  %q = getelementptr i32, ptr %base, i32 %j
  store i32 0, ptr %p
  store i32 0, ptr %q
  ret void
}

; CHECK-LABEL: Function: trunc_wrap
; CHECK: MayAlias: i32* %p, i32* %q
define void @trunc_wrap(ptr %base, i64 %x) {
  %next = add i64 %x, 4294967296
  %i = trunc i64 %x to i32
  %j = trunc i64 %next to i32
  %p = getelementptr i32, ptr %base, i32 %i
  %q = getelementptr i32, ptr %base, i32 %j
  store i32 0, ptr %p
  store i32 0, ptr %q
  ret void
}
