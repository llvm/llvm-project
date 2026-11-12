; RUN: opt < %s -loop-reduce
; PR6453

target datalayout = "e-p:64:64:64"

define void @_ZNK15PolynomialSpaceILi3EE13compute_indexEjRA3_j() nounwind {
entry:
  br label %bb6

bb6:
  %t4 = phi i32 [ 0, %entry ], [ %t3, %bb5 ]
  %t16 = sub i32 undef, %t4
  %t25 = sub i32 undef, %t4
  %t26 = add i32 undef, %t25
  br label %bb4

bb4:
  %t2 = phi i32 [ %t1, %bb3 ], [ 0, %bb6 ]
  %t17 = mul i32 %t2, %t16
  %t18 = zext i32 %t2 to i33
  %t19 = add i32 %t2, -1
  %t20 = zext i32 %t19 to i33
  %t21 = mul i33 %t18, %t20
  %t22 = lshr i33 %t21, 1
  %t23 = trunc i33 %t22 to i32
  %t24 = sub i32 %t17, %t23
  %t27 = add i32 %t24, %t26
  br i1 false, label %bb1, label %bb5

bb1:
  %t = icmp ugt i32 %t27, undef
  br i1 %t, label %bb2, label %bb3

bb3:
  %t1 = add i32 %t2, 1
  br label %bb4

bb5:
  %t3 = add i32 %t4, 1
  br label %bb6

bb2:
  ret void
}
