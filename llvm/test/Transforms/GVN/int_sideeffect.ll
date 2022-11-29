; RUN: opt -S < %s -passes=gvn | FileCheck %s

declare void @llvm.sideeffect()

; Store-to-load forwarding across a @llvm.sideeffect.

; CHECK-LABEL: s2l
; CHECK-NOT: load
define float @s2l(float* %p) {
    store float 0.0, float* %p
    call void @llvm.sideeffect()
    %t = load float, float* %p
    ret float %t
}

; Redundant load elimination across a @llvm.sideeffect.

; CHECK-LABEL: rle
; CHECK: load
; CHECK-NOT: load
define float @rle(float* %p) {
    %r = load float, float* %p
    call void @llvm.sideeffect()
    %s = load float, float* %p
    %t = fadd float %r, %s
    ret float %t
}

; LICM across a @llvm.sideeffect.

; CHECK-LABEL: licm
; CHECK: load
; CHECK: loop:
; CHECK-NOT: load
define float @licm(i64 %n, float* nocapture readonly %p) #0 {
bb0:
  br label %loop

loop:
  %i = phi i64 [ 0, %bb0 ], [ %t5, %loop ]
  %sum = phi float [ 0.000000e+00, %bb0 ], [ %t4, %loop ]
  call void @llvm.sideeffect()
  %t3 = load float, float* %p
  %t4 = fadd float %sum, %t3
  %t5 = add i64 %i, 1
  %t6 = icmp ult i64 %t5, %n
  br i1 %t6, label %loop, label %bb2

bb2:
  ret float %t4
}
