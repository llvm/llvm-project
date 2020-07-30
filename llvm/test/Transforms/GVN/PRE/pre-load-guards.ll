; RUN: opt < %s -basic-aa -gvn -enable-load-pre -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

declare void @llvm.experimental.guard(i1, ...)

; This is a motivating example on why we prohibit hoisting through guards.
; In the bottom block, we check that the index is within bounds and only access
; the element in this case and deoptimize otherwise. If we hoist the load to a
; place above the guard, it will may lead to out-of-bound array access.
define i32 @test_motivation(i32* %p, i32* %q, i1 %C, i32 %index, i32 %len) {
; CHECK-LABEL: @test_motivation(
block1:
  %el1 = getelementptr inbounds i32, i32* %q, i32 %index
  %el2 = getelementptr inbounds i32, i32* %p, i32 %index
	br i1 %C, label %block2, label %block3

block2:

; CHECK:        block2:
; CHECK-NEXT:     br
; CHECK-NOT:      load
; CHECK-NOT:      sge
; CHECK-NOT:      slt
; CHECK-NOT:      and
  br label %block4

block3:
  store i32 0, i32* %el1
  br label %block4

block4:

; CHECK:        block4:
; CHECK:          %cond1 = icmp sge i32 %index, 0
; CHECK-NEXT:     %cond2 = icmp slt i32 %index, %len
; CHECK-NEXT:     %in.bounds = and i1 %cond1, %cond2
; CHECK:          call void (i1, ...) @llvm.experimental.guard(i1 %in.bounds)
; CHECK-NEXT:     %PRE = load i32, i32* %P2
; CHECK:          ret i32 %PRE

  %P2 = phi i32* [%el2, %block3], [%el1, %block2]
  %cond1 = icmp sge i32 %index, 0
  %cond2 = icmp slt i32 %index, %len
  %in.bounds = and i1 %cond1, %cond2
  call void (i1, ...) @llvm.experimental.guard(i1 %in.bounds) [ "deopt"() ]
  %PRE = load i32, i32* %P2
  ret i32 %PRE
}

; Guard in load's block that is above the load should prohibit the PRE.
define i32 @test_guard_01(i32* %p, i32* %q, i1 %C, i1 %G) {
; CHECK-LABEL: @test_guard_01(
block1:
	br i1 %C, label %block2, label %block3

block2:

; CHECK:        block2:
; CHECK-NEXT:     br
; CHECK-NOT:      load

 br label %block4

block3:
  store i32 0, i32* %p
  br label %block4

block4:

; CHECK:        block4:
; CHECK:          call void (i1, ...) @llvm.experimental.guard(i1 %G)
; CHECK-NEXT:     load
; CHECK:          ret i32

  %P2 = phi i32* [%p, %block3], [%q, %block2]
  call void (i1, ...) @llvm.experimental.guard(i1 %G) [ "deopt"() ]
  %PRE = load i32, i32* %P2
  ret i32 %PRE
}

; Guard in load's block that is below the load should not prohibit the PRE.
define i32 @test_guard_02(i32* %p, i32* %q, i1 %C, i1 %G) {
; CHECK-LABEL: @test_guard_02(
block1:
	br i1 %C, label %block2, label %block3

block2:

; CHECK:        block2:
; CHECK-NEXT:     load i32, i32* %q

 br label %block4

block3:
  store i32 0, i32* %p
  br label %block4

block4:

; CHECK:        block4:
; CHECK-NEXT:     phi i32 [
; CHECK-NEXT:     phi i32* [
; CHECK-NEXT:     call void (i1, ...) @llvm.experimental.guard(i1 %G)
; CHECK-NOT:      load
; CHECK:          ret i32

  %P2 = phi i32* [%p, %block3], [%q, %block2]
  %PRE = load i32, i32* %P2
  call void (i1, ...) @llvm.experimental.guard(i1 %G) [ "deopt"() ]
  ret i32 %PRE
}

; Guard above the load's block should prevent PRE from hoisting through it.
define i32 @test_guard_03(i32* %p, i32* %q, i1 %C, i1 %G) {
; CHECK-LABEL: @test_guard_03(
block1:
	br i1 %C, label %block2, label %block3

block2:

; CHECK:        block2:
; CHECK-NEXT:     br
; CHECK-NOT:      load

 br label %block4

block3:
  store i32 0, i32* %p
  br label %block4

block4:

; CHECK:        block4:
; CHECK-NEXT:     phi i32*
; CHECK-NEXT:     call void (i1, ...) @llvm.experimental.guard(i1 %G)
; CHECK-NEXT:     load
; CHECK-NEXT:     ret i32

  %P2 = phi i32* [%p, %block3], [%q, %block2]
  call void (i1, ...) @llvm.experimental.guard(i1 %G) [ "deopt"() ]
  br label %block5

block5:
  %PRE = load i32, i32* %P2
  ret i32 %PRE
}
