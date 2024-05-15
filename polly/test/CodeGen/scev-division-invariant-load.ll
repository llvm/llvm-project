; RUN: opt %loadPolly -S -polly-codegen < %s
;
; Check that we generate valid code as we did not use the preloaded
; value of %tmp1 for the access function of the preloaded %tmp4.
;
; ModuleID = 'bug.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.frame_store = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr }
%struct.picture = type { i32, i32, i32, i32, i32, i32, [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32 }

define void @dpb_split_field(ptr %fs) {
entry:
  br label %for.body544

for.body544:                                      ; preds = %if.end908, %for.body544.lr.ph
  %indvars.iv87 = phi i64 [ 0, %entry ], [ %indvars.iv.next88, %if.end908 ]
  %tmp = phi ptr [ undef, %entry ], [ %tmp6, %if.end908 ]
  br label %land.lhs.true563

land.lhs.true563:                                 ; preds = %for.body544
  %size_x551 = getelementptr inbounds %struct.picture, ptr %tmp, i64 0, i32 18
  %tmp1 = load i32, ptr %size_x551, align 8
  %div552 = sdiv i32 %tmp1, 16
  %tmp2 = trunc i64 %indvars.iv87 to i32
  %div554 = sdiv i32 %tmp2, 4
  %mul555 = mul i32 %div552, %div554
  %tmp9 = add i32 %mul555, 0
  %tmp10 = shl i32 %tmp9, 1
  %add559 = add i32 %tmp10, 0
  %idxprom564 = sext i32 %add559 to i64
  %mb_field566 = getelementptr inbounds %struct.picture, ptr %tmp, i64 0, i32 31
  %tmp3 = load ptr, ptr %mb_field566, align 8
  %arrayidx567 = getelementptr inbounds i8, ptr %tmp3, i64 %idxprom564
  %tmp4 = load i8, ptr %arrayidx567, align 1
  %tobool569 = icmp eq i8 %tmp4, 0
  br i1 %tobool569, label %if.end908, label %if.then570

if.then570:                                       ; preds = %land.lhs.true563
  %frame = getelementptr inbounds %struct.frame_store, ptr %fs, i64 0, i32 10
  %tmp5 = load ptr, ptr %frame, align 8
  br label %if.end908

if.end908:                                        ; preds = %if.then570, %land.lhs.true563
  %tmp6 = phi ptr [ %tmp, %land.lhs.true563 ], [ undef, %if.then570 ]
  %indvars.iv.next88 = add nuw nsw i64 %indvars.iv87, 1
  br i1 undef, label %for.body544, label %for.inc912

for.inc912:                                       ; preds = %if.end908
  ret void
}
