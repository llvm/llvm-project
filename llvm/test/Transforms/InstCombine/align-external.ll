; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; Don't assume that external global variables or those with weak linkage have
; their preferred alignment. They may only have the ABI minimum alignment.

target datalayout = "i32:8:32"

@A = external global i32
@B = weak_odr global i32 0

@C = available_externally global <4 x i32> zeroinitializer, align 4
; CHECK: @C = available_externally global <4 x i32> zeroinitializer, align 4

define i64 @foo(i64 %a) {
  %t = ptrtoint ptr @A to i64
  %s = shl i64 %a, 3
  %r = or i64 %t, %s
  %q = add i64 %r, 1
  ret i64 %q
}

; CHECK-LABEL: define i64 @foo(i64 %a)
; CHECK: %s = shl i64 %a, 3
; CHECK: %r = or i64 %s, ptrtoint (ptr @A to i64)
; CHECK: %q = add i64 %r, 1
; CHECK: ret i64 %q

define i32 @bar() {
  %r = load i32, ptr @B, align 1
  ret i32 %r
}

; CHECK-LABEL: @bar()
; CHECK: align 1

define void @vec_store() {
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr @C, align 4
  ret void
}
; CHECK: define void @vec_store()
; CHECK: store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr @C, align 4
