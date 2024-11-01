; RUN: opt -opaque-pointers=0 %loadPolly -polly-codegen -S < %s \
; RUN:  -polly-invariant-load-hoisting \
; RUN:  | FileCheck %s

; CHECK: %polly.access.A = getelementptr float*, float** %A, i64 0
; CHECK: %polly.access.A.load = load float*, float** %polly.access.A
; CHECK: store float 4.200000e+01, float* %polly.access.A.load
; CHECK: store float 4.800000e+01, float* %polly.access.A.load

define void @foo(float** %A) {
start:
  br label %loop

loop:
  %indvar = phi i64 [0, %start], [%indvar.next, %latch]
  %indvar.next = add nsw i64 %indvar, 1
  %icmp = icmp slt i64 %indvar.next, 1024
  br i1 %icmp, label %body1, label %exit

body1:
  %baseA = load float*, float** %A
  store float 42.0, float* %baseA
  br label %body2

body2:
  %baseB = load float*, float** %A
  store float 48.0, float* %baseB
  br label %latch

latch:
  br label %loop

exit:
  ret void

}
