; REQUIRES: asserts
; RUN: opt < %s -instcombine -licm -loop-unswitch -enable-new-pm=0 -loop-unswitch-threshold=1000 -enable-mssa-loop-dependency=true -verify-memoryssa -disable-output -stats 2>&1| FileCheck %s
; Check no loop unswitch is done because unswitching of equality expr with
; undef is unsafe before the freeze patch is committed.
; CHECK-NOT: Number of branches unswitched

define void @ham(i64 %arg) local_unnamed_addr {
bb:
  %tmp = icmp eq i64 %arg, 0
  br i1 %tmp, label %bb3, label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = load volatile i64, i64* @global, align 8
  br label %bb3

bb3:                                              ; preds = %bb1, %bb
  %tmp4 = phi i64 [ %tmp2, %bb1 ], [ undef, %bb ]
  %tmp5 = load i64, i64* @global.1, align 8
  br label %bb6

bb6:                                              ; preds = %bb21, %bb3
  %tmp7 = phi i64 [ 3, %bb21 ], [ %tmp5, %bb3 ]
  %tmp8 = phi i64 [ %tmp25, %bb21 ], [ 0, %bb3 ]
  %tmp9 = icmp eq i64 %tmp7, %arg
  br i1 %tmp9, label %bb10, label %bb28

bb10:                                             ; preds = %bb6
  %tmp11 = icmp eq i64 %tmp7, 0
  br i1 %tmp11, label %bb21, label %bb12

bb12:                                             ; preds = %bb10
  %tmp13 = load i64, i64* @global.2, align 8
  %tmp14 = add nsw i64 %tmp13, 1
  store i64 %tmp14, i64* @global.2, align 8
  %tmp15 = load i64, i64* @global.3, align 8
  %tmp16 = icmp eq i64 %tmp15, %tmp4
  br i1 %tmp16, label %bb17, label %bb21

bb17:                                             ; preds = %bb12
  %tmp18 = phi i64 [ %tmp15, %bb12 ]
  %tmp19 = load i64, i64* @global.4, align 8
  %tmp20 = add nsw i64 %tmp19, %tmp18
  store i64 %tmp20, i64* @global.5, align 8
  br label %bb29

bb21:                                             ; preds = %bb12, %bb10
  %tmp22 = load i64, i64* @global.3, align 8
  %tmp23 = load volatile i64, i64* @global, align 8
  %tmp24 = add nsw i64 %tmp23, %tmp22
  store i64 %tmp24, i64* @global.5, align 8
  store i64 3, i64* @global.1, align 8
  %tmp25 = add nsw i64 %tmp8, 1
  %tmp26 = load i64, i64* @global.6, align 8
  %tmp27 = icmp slt i64 %tmp25, %tmp26
  br i1 %tmp27, label %bb6, label %bb28

bb28:                                             ; preds = %bb21, %bb6
  br label %bb29

bb29:                                             ; preds = %bb28, %bb17
  ret void
}

define void @zot(i64 %arg, i64 %arg1) local_unnamed_addr {
bb:
  %tmp = icmp eq i64 %arg, 0
  %tmp2 = select i1 %tmp, i64 %arg1, i64 undef
  %tmp3 = load i64, i64* @global.1, align 8
  br label %bb4

bb4:                                              ; preds = %bb19, %bb
  %tmp5 = phi i64 [ 3, %bb19 ], [ %tmp3, %bb ]
  %tmp6 = phi i64 [ %tmp23, %bb19 ], [ 0, %bb ]
  %tmp7 = icmp eq i64 %tmp5, %arg
  br i1 %tmp7, label %bb8, label %bb26

bb8:                                              ; preds = %bb4
  %tmp9 = icmp eq i64 %tmp5, 0
  br i1 %tmp9, label %bb19, label %bb10

bb10:                                             ; preds = %bb8
  %tmp11 = load i64, i64* @global.2, align 8
  %tmp12 = add nsw i64 %tmp11, 1
  store i64 %tmp12, i64* @global.2, align 8
  %tmp13 = load i64, i64* @global.3, align 8
  %tmp14 = icmp eq i64 %tmp13, %tmp2
  br i1 %tmp14, label %bb15, label %bb19

bb15:                                             ; preds = %bb10
  %tmp16 = phi i64 [ %tmp13, %bb10 ]
  %tmp17 = load i64, i64* @global.4, align 8
  %tmp18 = add nsw i64 %tmp17, %tmp16
  store i64 %tmp18, i64* @global.5, align 8
  br label %bb27

bb19:                                             ; preds = %bb10, %bb8
  %tmp20 = load i64, i64* @global.3, align 8
  %tmp21 = load volatile i64, i64* @global, align 8
  %tmp22 = add nsw i64 %tmp21, %tmp20
  store i64 %tmp22, i64* @global.5, align 8
  store i64 3, i64* @global.1, align 8
  %tmp23 = add nsw i64 %tmp6, 1
  %tmp24 = load i64, i64* @global.6, align 8
  %tmp25 = icmp slt i64 %tmp23, %tmp24
  br i1 %tmp25, label %bb4, label %bb26

bb26:                                             ; preds = %bb19, %bb4
  br label %bb27

bb27:                                             ; preds = %bb26, %bb15
  ret void
}

@global = common global i64 0, align 8
@global.1 = common global i64 0, align 8
@global.2 = common global i64 0, align 8
@global.3 = common global i64 0, align 8
@global.4 = common global i64 0, align 8
@global.5 = common global i64 0, align 8
@global.6 = common global i64 0, align 8


