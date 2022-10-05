; RUN: opt -S < %s -licm | FileCheck %s

declare void @llvm.sideeffect()

; LICM across a @llvm.sideeffect.

; CHECK-LABEL: licm
; CHECK: load
; CHECK: loop:
; CHECK-NOT: load
define float @licm(i64 %n, ptr nocapture readonly %p) #0 {
bb0:
  br label %loop

loop:
  %i = phi i64 [ 0, %bb0 ], [ %t5, %loop ]
  %sum = phi float [ 0.000000e+00, %bb0 ], [ %t4, %loop ]
  call void @llvm.sideeffect()
  %t3 = load float, ptr %p
  %t4 = fadd float %sum, %t3
  %t5 = add i64 %i, 1
  %t6 = icmp ult i64 %t5, %n
  br i1 %t6, label %loop, label %bb2

bb2:
  ret float %t4
}
