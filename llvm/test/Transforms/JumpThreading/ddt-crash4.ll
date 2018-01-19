; RUN: opt < %s -jump-threading -disable-output -verify-dom-info
@global = external global i64, align 8

define void @f() {
bb:
  br label %bb1

bb1:
  %tmp = load i64, i64* @global, align 8
  %tmp2 = icmp eq i64 %tmp, 0
  br i1 %tmp2, label %bb27, label %bb3

bb3:
  %tmp4 = load i64, i64* @global, align 8
  %tmp5 = icmp eq i64 %tmp4, 0
  br i1 %tmp5, label %bb6, label %bb7

bb6:
  br label %bb7

bb7:
  %tmp8 = phi i1 [ true, %bb3 ], [ undef, %bb6 ]
  %tmp9 = select i1 %tmp8, i64 %tmp4, i64 0
  br i1 false, label %bb10, label %bb23

bb10:
  %tmp11 = load i64, i64* @global, align 8
  %tmp12 = icmp slt i64 %tmp11, 5
  br i1 %tmp12, label %bb13, label %bb17

bb13:
  br label %bb14

bb14:
  br i1 undef, label %bb15, label %bb16

bb15:
  unreachable

bb16:
  br label %bb10

bb17:
  br label %bb18

bb18:
  br i1 undef, label %bb22, label %bb13

bb19:
  br i1 undef, label %bb20, label %bb21

bb20:
  unreachable

bb21:
  br label %bb18

bb22:
  br label %bb23

bb23:
  br i1 undef, label %bb24, label %bb13

bb24:
  br i1 undef, label %bb26, label %bb25

bb25:
  br label %bb19

bb26:
  br label %bb1

bb27:
  br label %bb24
}
