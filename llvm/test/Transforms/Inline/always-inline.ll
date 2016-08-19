; RUN: opt < %s -inline-threshold=0 -always-inline -S | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CALL
;
; Ensure the threshold has no impact on these decisions.
; RUN: opt < %s -inline-threshold=20000000 -always-inline -S | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CALL
; RUN: opt < %s -inline-threshold=-20000000 -always-inline -S | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CALL
;
; The new pass manager doesn't re-use any threshold based infrastructure for
; the always inliner, but test that we get the correct result.
; RUN: opt < %s -passes=always-inline -S | FileCheck %s --check-prefix=CHECK

define i32 @inner1() alwaysinline {
  ret i32 1
}
define i32 @outer1() {
; CHECK-LABEL: @outer1(
; CHECK-NOT: call
; CHECK: ret

   %r = call i32 @inner1()
   ret i32 %r
}

; The always inliner can't DCE internal functions. PR2945
; CHECK-LABEL: @pr2945(
define internal i32 @pr2945() nounwind {
  ret i32 0
}

define internal void @inner2(i32 %N) alwaysinline {
  %P = alloca i32, i32 %N
  ret void
}
define void @outer2(i32 %N) {
; The always inliner (unlike the normal one) should be willing to inline
; a function with a dynamic alloca into one without a dynamic alloca.
; rdar://6655932
;
; CHECK-LABEL: @outer2(
; CHECK-NOT: call void @inner2
; CHECK-NOT: call void @inner2
; CHECK: ret void

  call void @inner2( i32 %N )
  ret void
}

declare i32 @a() returns_twice
declare i32 @b() returns_twice

define i32 @inner3() alwaysinline {
entry:
  %call = call i32 @a() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}
define i32 @outer3() {
entry:
; CHECK-LABEL: @outer3(
; CHECK-NOT: call i32 @a
; CHECK: ret

  %call = call i32 @inner3()
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @inner4() alwaysinline returns_twice {
entry:
  %call = call i32 @b() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @outer4() {
entry:
; CHECK-LABEL: @outer4(
; CHECK: call i32 @b()
; CHECK: ret

  %call = call i32 @inner4() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @inner5(i8* %addr) alwaysinline {
entry:
  indirectbr i8* %addr, [ label %one, label %two ]

one:
  ret i32 42

two:
  ret i32 44
}
define i32 @outer5(i32 %x) {
; CHECK-LABEL: @outer5(
; CHECK: call i32 @inner5
; CHECK: ret

  %cmp = icmp slt i32 %x, 42
  %addr = select i1 %cmp, i8* blockaddress(@inner5, %one), i8* blockaddress(@inner5, %two)
  %call = call i32 @inner5(i8* %addr)
  ret i32 %call
}

define void @inner6(i32 %x) alwaysinline {
entry:
  %icmp = icmp slt i32 %x, 0
  br i1 %icmp, label %return, label %bb

bb:
  %sub = sub nsw i32 %x, 1
  call void @inner6(i32 %sub)
  ret void

return:
  ret void
}
define void @outer6() {
; CHECK-LABEL: @outer6(
; CHECK: call void @inner6(i32 42)
; CHECK: ret

entry:
  call void @inner6(i32 42)
  ret void
}

define i32 @inner7() {
  ret i32 1
}
define i32 @outer7() {
; CHECK-CALL-LABEL: @outer7(
; CHECK-CALL-NOT: call
; CHECK-CALL: ret

   %r = call i32 @inner7() alwaysinline
   ret i32 %r
}

define float* @inner8(float* nocapture align 128 %a) alwaysinline {
  ret float* %a
}
define float @outer8(float* nocapture %a) {
; CHECK-LABEL: @outer8(
; CHECK-NOT: call float* @inner8
; CHECK: ret

  %inner_a = call float* @inner8(float* %a)
  %f = load float, float* %inner_a, align 4
  ret float %f
}
