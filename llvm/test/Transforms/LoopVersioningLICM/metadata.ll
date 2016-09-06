; RUN: opt < %s  -O1  -S -loop-versioning-licm -licm 2>&1 | FileCheck %s

; CHECK-LABEL: @without_metadata(
define i32 @without_metadata(i32* nocapture %var1, i32* nocapture readnone %var2, i32* nocapture %var3, i32 %itr) #0 {
entry:
  %cmp14 = icmp eq i32 %itr, 0
  br i1 %cmp14, label %for.end13, label %for.cond1.preheader.preheader

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc11
  %j.016 = phi i32 [ %j.1.lcssa, %for.inc11 ], [ 0, %for.cond1.preheader.preheader ]
  %i.015 = phi i32 [ %inc12, %for.inc11 ], [ 0, %for.cond1.preheader.preheader ]
  %cmp212 = icmp ult i32 %j.016, %itr
  br i1 %cmp212, label %for.body3.lr.ph, label %for.inc11

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %add = add i32 %i.015, %itr
  %idxprom6 = zext i32 %i.015 to i64
  %arrayidx7 = getelementptr inbounds i32, i32* %var3, i64 %idxprom6
  br label %for.body3

for.body3:                                        ; preds = %for.body3.lr.ph, %for.body3
  %j.113 = phi i32 [ %j.016, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %idxprom = zext i32 %j.113 to i64
  %arrayidx = getelementptr inbounds i32, i32* %var1, i64 %idxprom
; CHECK: store i32 %add, i32* %arrayidx, align 4, !alias.scope {{.*}}, !noalias {{.*}}
  store i32 %add, i32* %arrayidx, align 4
  %0 = load i32, i32* %arrayidx7, align 4
  %add8 = add nsw i32 %0, %add
  store i32 %add8, i32* %arrayidx7, align 4
  %inc = add nuw i32 %j.113, 1
  %cmp2 = icmp ult i32 %inc, %itr
  br i1 %cmp2, label %for.body3, label %for.inc11.loopexit

for.inc11.loopexit:                               ; preds = %for.body3
  br label %for.inc11

for.inc11:                                        ; preds = %for.inc11.loopexit, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.016, %for.cond1.preheader ], [ %itr, %for.inc11.loopexit ]
  %inc12 = add nuw i32 %i.015, 1
  %cmp = icmp ult i32 %inc12, %itr
  br i1 %cmp, label %for.cond1.preheader, label %for.end13.loopexit

for.end13.loopexit:                               ; preds = %for.inc11
  br label %for.end13

for.end13:                                        ; preds = %for.end13.loopexit, %entry
  ret i32 0
}

; CHECK-LABEL: @with_metadata(
define i32 @with_metadata(i32* nocapture %var1, i32* nocapture readnone %var2, i32* nocapture %var3, i32 %itr) #0 {
entry:
  %cmp14 = icmp eq i32 %itr, 0
  br i1 %cmp14, label %for.end13, label %for.cond1.preheader.preheader

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc11
  %j.016 = phi i32 [ %j.1.lcssa, %for.inc11 ], [ 0, %for.cond1.preheader.preheader ]
  %i.015 = phi i32 [ %inc12, %for.inc11 ], [ 0, %for.cond1.preheader.preheader ]
  %cmp212 = icmp ult i32 %j.016, %itr
  br i1 %cmp212, label %for.body3.lr.ph, label %for.inc11

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %add = add i32 %i.015, %itr
  %idxprom6 = zext i32 %i.015 to i64
  %arrayidx7 = getelementptr inbounds i32, i32* %var3, i64 %idxprom6
  br label %for.body3

for.body3:                                        ; preds = %for.body3.lr.ph, %for.body3
  %j.113 = phi i32 [ %j.016, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %idxprom = zext i32 %j.113 to i64
  %arrayidx = getelementptr inbounds i32, i32* %var1, i64 %idxprom
; CHECK-NOT: store i32 %add, i32* %arrayidx, align 4, !alias.scope {{.*}}, !noalias {{.*}}
  store i32 %add, i32* %arrayidx, align 4
  %0 = load i32, i32* %arrayidx7, align 4
  %add8 = add nsw i32 %0, %add
  store i32 %add8, i32* %arrayidx7, align 4
  %inc = add nuw i32 %j.113, 1
  %cmp2 = icmp ult i32 %inc, %itr
  br i1 %cmp2, label %for.body3, label %for.inc11.loopexit, !llvm.loop !0

for.inc11.loopexit:                               ; preds = %for.body3
  br label %for.inc11

for.inc11:                                        ; preds = %for.inc11.loopexit, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.016, %for.cond1.preheader ], [ %itr, %for.inc11.loopexit ]
  %inc12 = add nuw i32 %i.015, 1
  %cmp = icmp ult i32 %inc12, %itr
  br i1 %cmp, label %for.cond1.preheader, label %for.end13.loopexit

for.end13.loopexit:                               ; preds = %for.inc11
  br label %for.end13

for.end13:                                        ; preds = %for.end13.loopexit, %entry
  ret i32 0
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.licm_versioning.disable"}
