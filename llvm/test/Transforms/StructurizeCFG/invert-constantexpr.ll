; RUN: opt -S -o - -structurizecfg < %s | FileCheck %s

; CHECK-LABEL: @invert_constantexpr_condition(
; CHECK: %tmp5 = or i1 %tmp4, icmp eq (i32 bitcast (float fadd (float undef, float undef) to i32), i32 0)
; CHECK: [ icmp ne (i32 bitcast (float fadd (float undef, float undef) to i32), i32 0), %bb ]
define void @invert_constantexpr_condition(i32 %arg, i32 %arg1) #0 {
bb:
  %tmp = icmp eq i32 %arg, 0
  br i1 icmp eq (i32 bitcast (float fadd (float undef, float undef) to i32), i32 0), label %bb2, label %bb6

bb2:
  br i1 %tmp, label %bb3, label %bb6

bb3:
  %tmp4 = phi i1 [ %tmp7, %bb6 ], [ undef, %bb2 ]
  %tmp5 = or i1 %tmp4, icmp eq (i32 bitcast (float fadd (float undef, float undef) to i32), i32 0)
  br i1 %tmp5, label %bb8, label %bb8

bb6:
  %tmp7 = icmp slt i32 %arg, %arg1
  br label %bb3

bb8:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
