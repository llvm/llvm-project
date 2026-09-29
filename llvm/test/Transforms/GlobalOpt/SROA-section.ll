; Verify that section assignment is copied during SROA
; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK: @G.0
; CHECK: section ".foo"
; CHECK: @G.1
; CHECK: section ".foo"
; CHECK: @G.2
; CHECK: section ".foo"

%T = type { double, double, double }
@G = internal global %T zeroinitializer, align 16, section ".foo"

define void @test() {
  store double 1.0, ptr @G, align 16
  store double 2.0, ptr getelementptr (%T, ptr @G, i32 0, i32 1), align 8
  store double 3.0, ptr getelementptr (%T, ptr @G, i32 0, i32 2), align 16
  ret void
}

define double @test2() {
  %V1 = load double, ptr @G, align 16
  %V2 = load double, ptr getelementptr (%T, ptr @G, i32 0, i32 1), align 8
  %V3 = load double, ptr getelementptr (%T, ptr @G, i32 0, i32 2), align 16
  %R = fadd double %V1, %V2
  %R2 = fadd double %R, %V3
  ret double %R2
}
