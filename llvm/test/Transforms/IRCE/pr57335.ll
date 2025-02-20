; RUN: opt -passes=irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -passes='require<branch-prob>,irce' -S < %s 2>&1 | FileCheck %s

; Make sure we don't crash.
define void @test(i1 %arg) {
; CHECK-LABEL: test
bb:
  %tmp = icmp ult i32 0, undef
  br i1 %tmp, label %bb1, label %bb31

bb1:                                              ; preds = %bb
  br label %bb5

bb2:                                              ; preds = %bb20
  %tmp3 = phi i32 [ %tmp21, %bb20 ]
  %tmp4 = add nuw nsw i32 %tmp3, 100
  br label %bb5

bb5:                                              ; preds = %bb2, %bb1
  %tmp6 = phi i32 [ 102, %bb1 ], [ %tmp4, %bb2 ]
  %tmp7 = phi i32 [ 2, %bb1 ], [ %tmp3, %bb2 ]
  br label %bb12

bb8:                                              ; preds = %bb12
  %tmp9 = phi i32 [ %tmp14, %bb12 ]
  %tmp10 = add nsw i32 %tmp9, -1
  %tmp11 = icmp ult i32 %tmp10, undef
  br i1 %tmp11, label %bb12, label %bb32

bb12:                                             ; preds = %bb8, %bb5
  %tmp13 = phi i32 [ 1, %bb5 ], [ %tmp9, %bb8 ]
  %tmp14 = add nuw nsw i32 %tmp13, 1
  %tmp15 = load atomic i32, ptr addrspace(1) undef unordered, align 8
  %tmp16 = icmp ult i32 %tmp14, %tmp6
  br i1 %tmp16, label %bb8, label %bb17

bb17:                                             ; preds = %bb12
  %tmp18 = phi i32 [ %tmp15, %bb12 ]
  %tmp19 = icmp ult i32 %tmp7, %tmp18
  br i1 %tmp19, label %bb20, label %bb33

bb20:                                             ; preds = %bb17
  %tmp21 = add nuw nsw i32 %tmp7, 2
  br i1 %arg, label %bb22, label %bb2

bb22:                                             ; preds = %bb20
  %tmp23 = phi i32 [ %tmp18, %bb20 ]
  br label %bb24

bb24:                                             ; preds = %bb30, %bb22
  %tmp25 = phi i32 [ 3, %bb22 ], [ %tmp26, %bb30 ]
  %tmp26 = add i32 %tmp25, 1
  %tmp27 = icmp ult i32 %tmp26, %tmp23
  %tmp28 = and i1 undef, %tmp27
  br i1 %tmp28, label %bb30, label %bb29

bb29:                                             ; preds = %bb24
  ret void

bb30:                                             ; preds = %bb24
  br label %bb24

bb31:                                             ; preds = %bb
  ret void

bb32:                                             ; preds = %bb8
  unreachable

bb33:                                             ; preds = %bb17
  ret void
}
