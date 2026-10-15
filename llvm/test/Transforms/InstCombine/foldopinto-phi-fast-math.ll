; RUN: opt < %s -passes=instcombine -S | FileCheck %s

declare void @sink(float)

define i1 @test(i1 %c, float %x) {
entry:
  br i1 %c, label %a, label %b
a:
  br label %m
b:
  br label %m
m:
  %p = phi float [ %x, %a ], [ 2.0, %b ]
  %u1 = fadd nnan float %p, 1.0
  call void @sink(float %u1)
  %u2 = fadd float %p, 1.0
  call void @sink(float %u2)
  ret i1 %c
}

; CHECK-LABEL: define i1 @test(
; CHECK: %[[NNAN:.*]] = fadd nnan float %p, 1.000000e+00
; CHECK: call void @sink(float %[[NNAN]])
; CHECK: %[[PLAIN:.*]] = fadd float %p, 1.000000e+00
; CHECK: call void @sink(float %[[PLAIN]])
