; RUN: opt -S -debug-only=iroutliner -p=iroutliner -ir-outlining-no-cost %s -o - 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK-NOT: ... Skipping function with nooutline attribute: outlinable
; CHECK-NOT: @outlined_ir_func
; CHECK: ... Skipping function with nooutline attribute: nooutline1
; CHECK: ... Skipping function with nooutline attribute: nooutline2

define void @outlinable() { ret void }

define i8 @nooutline1(ptr noalias %s, ptr noalias %d, i64 %len) "nooutline" {
  %a = load i8, ptr %s
  %b = load i8, ptr %d
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr %d, ptr %s, i64 %len, i1 false)
  %c = add i8 %a, %b
  %ret = load i8, ptr %s
  ret i8 %ret
}

define i8 @nooutline2(ptr noalias %s, ptr noalias %d, i64 %len) "nooutline" {
  %a = load i8, ptr %s
  %b = load i8, ptr %d
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr %d, ptr %s, i64 %len, i1 false)
  %c = add i8 %a, %b
  %ret = load i8, ptr %s
  ret i8 %ret
}

declare void @llvm.memcpy.p0i8.p0i8.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

