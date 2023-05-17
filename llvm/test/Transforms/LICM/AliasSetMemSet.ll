; RUN: opt < %s -passes='loop-mssa(loop-deletion,licm,loop-idiom)' -disable-output
; Check no assertion when loop-idiom deletes the MemSet already analyzed by licm
define void @set_array() {
  br i1 false, label %bb3.preheader.lr.ph, label %bb9

bb3.preheader.lr.ph:                              ; preds = %0
  br label %bb3.preheader

bb4:                                              ; preds = %bb4.lr.ph, %bb7
  %j.3.06 = phi i8 [ %j.3.17, %bb4.lr.ph ], [ %_tmp13, %bb7 ]
  br label %bb6

bb6:                                              ; preds = %bb4, %bb6
  %k.4.04 = phi i8 [ 0, %bb4 ], [ %_tmp9, %bb6 ]
  %_tmp31 = sext i8 %j.3.06 to i64
  %_tmp4 = mul i64 %_tmp31, 10
  %_tmp5 = getelementptr i8, ptr undef, i64 %_tmp4
  %_tmp7 = getelementptr i8, ptr %_tmp5, i8 %k.4.04
  store i8 42, ptr %_tmp7
  %_tmp9 = add i8 %k.4.04, 1
  %_tmp11 = icmp slt i8 %_tmp9, 10
  br i1 %_tmp11, label %bb6, label %bb7

bb7:                                              ; preds = %bb6
  %_tmp13 = add i8 %j.3.06, 1
  %_tmp15 = icmp slt i8 %_tmp13, 2
  br i1 %_tmp15, label %bb4, label %bb3.bb1.loopexit_crit_edge

bb3.bb1.loopexit_crit_edge:                       ; preds = %bb7
  %split = phi i8 [ %_tmp13, %bb7 ]
  br label %bb1.loopexit

bb1.loopexit:                                     ; preds = %bb3.bb1.loopexit_crit_edge, %bb3.preheader
  %j.3.0.lcssa = phi i8 [ %split, %bb3.bb1.loopexit_crit_edge ], [ %j.3.17, %bb3.preheader ]
  br i1 false, label %bb3.preheader, label %bb1.bb9_crit_edge

bb3.preheader:                                    ; preds = %bb3.preheader.lr.ph, %bb1.loopexit
  %j.3.17 = phi i8 [ undef, %bb3.preheader.lr.ph ], [ %j.3.0.lcssa, %bb1.loopexit ]
  %_tmp155 = icmp slt i8 %j.3.17, 2
  br i1 %_tmp155, label %bb4.lr.ph, label %bb1.loopexit

bb4.lr.ph:                                        ; preds = %bb3.preheader
  br label %bb4

bb1.bb9_crit_edge:                                ; preds = %bb1.loopexit
  br label %bb9

bb9:                                              ; preds = %bb1.bb9_crit_edge, %0
  ret void
}

