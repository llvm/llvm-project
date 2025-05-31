; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s \
; RUN:  -polly-invariant-load-hoisting \
; RUN:  | FileCheck %s

; CHECK: %polly.access.A = getelementptr ptr, ptr %A, i64 0
; CHECK: %polly.access.A.load = load ptr, ptr %polly.access.A
; CHECK: store float 4.200000e+01, ptr %polly.access.A.load
; CHECK: store float 4.800000e+01, ptr %polly.access.A.load

define void @foo(ptr %A) {
start:
  br label %loop

loop:
  %indvar = phi i64 [0, %start], [%indvar.next, %latch]
  %indvar.next = add nsw i64 %indvar, 1
  %icmp = icmp slt i64 %indvar.next, 1024
  br i1 %icmp, label %body1, label %exit

body1:
  %baseA = load ptr, ptr %A
  store float 42.0, ptr %baseA
  br label %body2

body2:
  %baseB = load ptr, ptr %A
  store float 48.0, ptr %baseB
  br label %latch

latch:
  br label %loop

exit:
  ret void

}
