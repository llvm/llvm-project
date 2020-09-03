; RUN: opt < %s -O2 -inlinedefault-threshold=100 -S | FileCheck %s

; Check that the inlinedefault-threshold does not alter the inline threshold
; for optsize or minsize functions

@a = global i32 4

define i32 @inner() {
  call void @extern()
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a1
  ret i32 %x1
}

define i32 @inner2() {
  call void @extern()
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32, i32* @a
  %x2 = add i32 %x1, %a2
  %a3 = load volatile i32, i32* @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32, i32* @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32, i32* @a
  %x5 = add i32 %x3, %a5
  %a6 = load volatile i32, i32* @a
  %x6 = add i32 %x5, %a6
  %a7 = load volatile i32, i32* @a
  %x7 = add i32 %x6, %a7
  %a8 = load volatile i32, i32* @a
  %x8 = add i32 %x7, %a8
  ret i32 %x8
}

define i32 @inner3() {
  call void @extern()
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32, i32* @a
  %x2 = add i32 %x1, %a2
  %a3 = load volatile i32, i32* @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32, i32* @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32, i32* @a
  %x5 = add i32 %x4, %a5
  %a6 = load volatile i32, i32* @a
  %x6 = add i32 %x5, %a6
  %a7 = load volatile i32, i32* @a
  %x7 = add i32 %x6, %a7
  %a8 = load volatile i32, i32* @a
  %x8 = add i32 %x7, %a8
  %a9 = load volatile i32, i32* @a
  %x9 = add i32 %x8, %a9
  %a10 = load volatile i32, i32* @a
  %x10 = add i32 %x9, %a10
  %a11 = load volatile i32, i32* @a
  %x11 = add i32 %x10, %a11
  %a12 = load volatile i32, i32* @a
  %x12 = add i32 %x11, %a12
  %a13 = load volatile i32, i32* @a
  %x13 = add i32 %x12, %a13
  %a14 = load volatile i32, i32* @a
  %x14 = add i32 %x13, %a14
  %a15 = load volatile i32, i32* @a
  %x15 = add i32 %x14, %a15
  ret i32 %x15
}

define i32 @outer() optsize {
; CHECK-LABEL: @outer
; CHECK-NOT: call i32 @inner()
   %r = call i32 @inner()
   ret i32 %r
}

define i32 @outer2() optsize {
; CHECK-LABEL: @outer2
; CHECK: call i32 @inner2()
   %r = call i32 @inner2()
   ret i32 %r
}

define i32 @outer3() minsize {
; CHECK-LABEL: @outer3
; CHECK: call i32 @inner()
   %r = call i32 @inner()
   ret i32 %r
}

define i32 @outer4() {
; CHECK-LABEL: @outer4
; CHECK-NOT: call i32 @inner()
   %r = call i32 @inner()
   ret i32 %r
}

define i32 @outer5() {
; CHECK-LABEL: @outer5
; CHECK-NOT: call i32 @inner2()
   %r = call i32 @inner2()
   ret i32 %r
}

define i32 @outer6() {
; CHECK-LABEL: @outer6
; CHECK: call i32 @inner3()
   %r = call i32 @inner3()
   ret i32 %r
}

declare void @extern()
