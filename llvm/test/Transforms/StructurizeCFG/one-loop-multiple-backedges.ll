; RUN: opt -S -structurizecfg %s -o - | FileCheck %s

; CHECK-NOT: br i1 true

define void @blam(i32 addrspace(1)* nocapture %arg, float %arg1, float %arg2) {
; CHECK: bb:
bb:
  br label %bb3

; CHECK: bb3:
bb3:                                              ; preds = %bb7, %bb
  %tmp = phi i64 [ 0, %bb ], [ %tmp8, %bb7 ]
  %tmp4 = fcmp ult float %arg1, 3.500000e+00
; CHECK: %0 = xor i1 %tmp4, true
; CHECK: br i1 %0, label %bb5, label %Flow
  br i1 %tmp4, label %bb7, label %bb5

; CHECK: bb5:
bb5:                                              ; preds = %bb3
  %tmp6 = fcmp olt float 0.000000e+00, %arg2
; CHECK: br label %Flow
  br i1 %tmp6, label %bb10, label %bb7

; CHECK: Flow:
; CHECK: br i1 %3, label %bb7, label %Flow1

; CHECK: bb7
bb7:                                              ; preds = %bb5, %bb3
  %tmp8 = add nuw nsw i64 %tmp, 1
  %tmp9 = icmp slt i64 %tmp8, 5
; CHECK: br label %Flow1
  br i1 %tmp9, label %bb3, label %bb10

; CHECK: Flow1:
; CHECK: br i1 %7, label %bb10, label %bb3

; CHECK: bb10
bb10:                                             ; preds = %bb7, %bb5
  %tmp11 = phi i32 [ 15, %bb5 ], [ 255, %bb7 ]
  store i32 %tmp11, i32 addrspace(1)* %arg, align 4
  ret void
}
