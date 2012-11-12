; RUN: opt < %s -instcombine -S | FileCheck %s

; Don't assume that external global variables or those with weak linkage have
; their preferred alignment. They may only have the ABI minimum alignment.

; CHECK: %s = shl i64 %a, 3
; CHECK: %r = or i64 %s, ptrtoint (i32* @A to i64)
; CHECK: %q = add i64 %r, 1
; CHECK: ret i64 %q

target datalayout = "-i32:8:32"

@A = external global i32
@B = weak_odr global i32 0

define i64 @foo(i64 %a) {
  %t = ptrtoint i32* @A to i64
  %s = shl i64 %a, 3
  %r = or i64 %t, %s
  %q = add i64 %r, 1
  ret i64 %q
}

define i32 @bar() {
; CHECK: @bar
  %r = load i32* @B, align 1
; CHECK: align 1
  ret i32 %r
}
