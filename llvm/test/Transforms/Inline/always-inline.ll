; RUN: opt < %s -inline-threshold=0 -always-inline -enable-new-pm=0 -S | FileCheck %s --check-prefix=CHECK
;
; Ensure the threshold has no impact on these decisions.
; RUN: opt < %s -inline-threshold=20000000 -always-inline -enable-new-pm=0 -S | FileCheck %s --check-prefix=CHECK
; RUN: opt < %s -inline-threshold=-20000000 -always-inline -enable-new-pm=0 -S | FileCheck %s --check-prefix=CHECK
;
; The new pass manager doesn't re-use any threshold based infrastructure for
; the always inliner, but test that we get the correct result.
; RUN: opt < %s -inline-threshold=0 -passes=always-inline -S | FileCheck %s --check-prefix=CHECK
; RUN: opt < %s -inline-threshold=20000000 -passes=always-inline -S | FileCheck %s --check-prefix=CHECK
; RUN: opt < %s -inline-threshold=-20000000 -passes=always-inline -S | FileCheck %s --check-prefix=CHECK

define internal i32 @inner1() alwaysinline {
; CHECK-NOT: @inner1(
  ret i32 1
}
define i32 @outer1() {
; CHECK-LABEL: @outer1(
; CHECK-NOT: call
; CHECK: ret

   %r = call i32 @inner1()
   ret i32 %r
}

; The always inliner can't DCE arbitrary internal functions. PR2945
define internal i32 @pr2945() nounwind {
; CHECK-LABEL: @pr2945(
  ret i32 0
}

define internal void @inner2(i32 %N) alwaysinline {
; CHECK-NOT: @inner2(
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
; CHECK: ret void

  call void @inner2( i32 %N )
  ret void
}

declare i32 @a() returns_twice
declare i32 @b() returns_twice

; Cannot alwaysinline when that would introduce a returns_twice call.
define internal i32 @inner3() alwaysinline {
; CHECK-LABEL: @inner3(
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

define internal i32 @inner4() alwaysinline returns_twice {
; CHECK-NOT: @inner4(
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

; We can't inline this even though it has alwaysinline!
define internal i32 @inner5(i8* %addr) alwaysinline {
; CHECK-LABEL: @inner5(
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

; We never inline a function that calls itself recursively.
define internal void @inner6(i32 %x) alwaysinline {
; CHECK-LABEL: @inner6(
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

; This is not an alwaysinline function and is actually external.
define i32 @inner7() {
; CHECK-LABEL: @inner7(
  ret i32 1
}
define i32 @outer7() {
; CHECK-LABEL: @outer7(
; CHECK-NOT: call
; CHECK: ret
   %r = call i32 @inner7() alwaysinline
   ret i32 %r
}

define internal float* @inner8(float* nocapture align 128 %a) alwaysinline {
; CHECK-NOT: @inner8(
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


; The 'inner9*' and 'outer9' functions are designed to check that we remove
; a function that is inlined by the always inliner even when it is used by
; a complex constant expression prior to being inlined.

; The 'a' function gets used in a complex constant expression that, despite
; being constant folded, means it isn't dead. As a consequence it shouldn't be
; deleted. If it is, then the constant expression needs to become more complex
; to accurately test this scenario.
define internal void @inner9a(i1 %b) alwaysinline {
; CHECK-LABEL: @inner9a(
entry:
  ret void
}

define internal void @inner9b(i1 %b) alwaysinline {
; CHECK-NOT: @inner9b(
entry:
  ret void
}

declare void @dummy9(i1 %b)

define void @outer9() {
; CHECK-LABEL: @outer9(
entry:
  ; First we use @inner9a in a complex constant expression that may get folded
  ; but won't get removed, and then we call it which will get inlined. Despite
  ; this the function can't be deleted because of the constant expression
  ; usage.
  %sink = alloca i1
  store volatile i1 icmp eq (i64 ptrtoint (void (i1)* @inner9a to i64), i64 ptrtoint(void (i1)* @dummy9 to i64)), i1* %sink
; CHECK: store volatile
  call void @inner9a(i1 false)
; CHECK-NOT: call void @inner9a

  ; Next we call @inner9b passing in a constant expression. This constant
  ; expression will in fact be removed by inlining, so we should also be able
  ; to delete the function.
  call void @inner9b(i1 icmp eq (i64 ptrtoint (void (i1)* @inner9b to i64), i64 ptrtoint(void (i1)* @dummy9 to i64)))
; CHECK-NOT: @inner9b

  ret void
; CHECK: ret void
}

; The 'inner10' and 'outer10' functions test a frustrating consequence of the
; current 'alwaysinline' semantic model. Because such functions are allowed to
; be external functions, it may be necessary to both inline all of their uses
; and leave them in the final output. These tests can be removed if and when
; we restrict alwaysinline further.
define void @inner10() alwaysinline {
; CHECK-LABEL: @inner10(
entry:
  ret void
}

define void @outer10() {
; CHECK-LABEL: @outer10(
entry:
  call void @inner10()
; CHECK-NOT: call void @inner10

  ret void
; CHECK: ret void
}

; The 'inner11' and 'outer11' functions test another dimension of non-internal
; functions with alwaysinline. These functions use external linkages that we can
; actually remove safely and so we should.
define linkonce void @inner11a() alwaysinline {
; CHECK-NOT: @inner11a(
entry:
  ret void
}

define available_externally void @inner11b() alwaysinline {
; CHECK-NOT: @inner11b(
entry:
  ret void
}

define void @outer11() {
; CHECK-LABEL: @outer11(
entry:
  call void @inner11a()
  call void @inner11b()
; CHECK-NOT: call void @inner11a
; CHECK-NOT: call void @inner11b

  ret void
; CHECK: ret void
}

; The 'inner12' and 'outer12' functions test that we don't remove functions
; which are part of a comdat group even if they otherwise seem dead.
$comdat12 = comdat any

define linkonce void @inner12() alwaysinline comdat($comdat12) {
; CHECK-LABEL: @inner12(
  ret void
}

define void @outer12() comdat($comdat12) {
; CHECK-LABEL: @outer12(
entry:
  call void @inner12()
; CHECK-NOT: call void @inner12

  ret void
; CHECK: ret void
}

; The 'inner13*' and 'outer13' functions test that we do remove functions
; which are part of a comdat group where all of the members are removed during
; always inlining.
$comdat13 = comdat any

define linkonce void @inner13a() alwaysinline comdat($comdat13) {
; CHECK-NOT: @inner13a(
  ret void
}

define linkonce void @inner13b() alwaysinline comdat($comdat13) {
; CHECK-NOT: @inner13b(
  ret void
}

define void @outer13() {
; CHECK-LABEL: @outer13(
entry:
  call void @inner13a()
  call void @inner13b()
; CHECK-NOT: call void @inner13a
; CHECK-NOT: call void @inner13b

  ret void
; CHECK: ret void
}

define void @inner14() readnone nounwind {
; CHECK: define void @inner14
  ret void
}

define void @outer14() {
; CHECK: call void @inner14
  call void @inner14()
  ret void
}

define internal i32 @inner15() {
; CHECK: @inner15(
  ret i32 1
}

define i32 @outer15() {
; CHECK-LABEL: @outer15(
; CHECK: call

   %r = call i32 @inner15() noinline
   ret i32 %r
}

define internal i32 @inner16() alwaysinline {
; CHECK: @inner16(
  ret i32 1
}

define i32 @outer16() {
; CHECK-LABEL: @outer16(
; CHECK: call

   %r = call i32 @inner16() noinline
   ret i32 %r
}

define i32 @inner17() alwaysinline {
; CHECK: @inner17(
  ret i32 1
}

define i32 @outer17() {
; CHECK-LABEL: @outer17(
; CHECK: call

   %r = call i32 @inner17() noinline
   ret i32 %r
}

define i32 @inner18() noinline {
; CHECK: @inner18(
  ret i32 1
}

define i32 @outer18() {
; CHECK-LABEL: @outer18(
; CHECK-NOT: call
; CHECK: ret

   %r = call i32 @inner18() alwaysinline

   ret i32 %r
}
