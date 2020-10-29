; RUN: opt -S -loop-versioning-licm -licm-versioning-invariant-threshold=0 %s | FileCheck %s
; RUN: opt -S -passes='loop-versioning-licm' -licm-versioning-invariant-threshold=0 %s | FileCheck %s

; Make sure the convergent attribute is respected, and no condition is
; introduced

; CHECK-LABEL: @test_convergent(
; CHECK: call void @llvm.convergent()
; CHECK-NOT: call void @llvm.convergent()
define i32 @test_convergent(i32* nocapture %var1, i32* nocapture readnone %var2, i32* nocapture %var3, i32 %itr) #1 {
entry:
  %cmp14 = icmp eq i32 %itr, 0
  br i1 %cmp14, label %for.end13, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc11
  %j.016 = phi i32 [ %j.1.lcssa, %for.inc11 ], [ 0, %entry ]
  %i.015 = phi i32 [ %inc12, %for.inc11 ], [ 0, %entry ]
  %cmp212 = icmp ult i32 %j.016, %itr
  br i1 %cmp212, label %for.body3.lr.ph, label %for.inc11

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %add = add i32 %i.015, %itr
  %idxprom6 = zext i32 %i.015 to i64
  %arrayidx7 = getelementptr inbounds i32, i32* %var3, i64 %idxprom6
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %j.113 = phi i32 [ %j.016, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %idxprom = zext i32 %j.113 to i64
  %arrayidx = getelementptr inbounds i32, i32* %var1, i64 %idxprom
  store i32 %add, i32* %arrayidx, align 4
  %load.arrayidx7 = load i32, i32* %arrayidx7, align 4
  call void @llvm.convergent()
  %add8 = add nsw i32 %load.arrayidx7, %add
  store i32 %add8, i32* %arrayidx7, align 4
  %inc = add nuw i32 %j.113, 1
  %cmp2 = icmp ult i32 %inc, %itr
  br i1 %cmp2, label %for.body3, label %for.inc11

for.inc11:                                        ; preds = %for.body3, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.016, %for.cond1.preheader ], [ %itr, %for.body3 ]
  %inc12 = add nuw i32 %i.015, 1
  %cmp = icmp ult i32 %inc12, %itr
  br i1 %cmp, label %for.cond1.preheader, label %for.end13

for.end13:                                        ; preds = %for.inc11, %entry
  ret i32 0
}

; CHECK-LABEL: @test_noduplicate(
; CHECK: call void @llvm.noduplicate()
; CHECK-NOT: call void @llvm.noduplicate()
define i32 @test_noduplicate(i32* nocapture %var1, i32* nocapture readnone %var2, i32* nocapture %var3, i32 %itr) #2 {
entry:
  %cmp14 = icmp eq i32 %itr, 0
  br i1 %cmp14, label %for.end13, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc11
  %j.016 = phi i32 [ %j.1.lcssa, %for.inc11 ], [ 0, %entry ]
  %i.015 = phi i32 [ %inc12, %for.inc11 ], [ 0, %entry ]
  %cmp212 = icmp ult i32 %j.016, %itr
  br i1 %cmp212, label %for.body3.lr.ph, label %for.inc11

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %add = add i32 %i.015, %itr
  %idxprom6 = zext i32 %i.015 to i64
  %arrayidx7 = getelementptr inbounds i32, i32* %var3, i64 %idxprom6
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %j.113 = phi i32 [ %j.016, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %idxprom = zext i32 %j.113 to i64
  %arrayidx = getelementptr inbounds i32, i32* %var1, i64 %idxprom
  store i32 %add, i32* %arrayidx, align 4
  %load.arrayidx7 = load i32, i32* %arrayidx7, align 4
  call void @llvm.noduplicate()
  %add8 = add nsw i32 %load.arrayidx7, %add
  store i32 %add8, i32* %arrayidx7, align 4
  %inc = add nuw i32 %j.113, 1
  %cmp2 = icmp ult i32 %inc, %itr
  br i1 %cmp2, label %for.body3, label %for.inc11

for.inc11:                                        ; preds = %for.body3, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.016, %for.cond1.preheader ], [ %itr, %for.body3 ]
  %inc12 = add nuw i32 %i.015, 1
  %cmp = icmp ult i32 %inc12, %itr
  br i1 %cmp, label %for.cond1.preheader, label %for.end13

for.end13:                                        ; preds = %for.inc11, %entry
  ret i32 0
}

declare void @llvm.convergent() #1
declare void @llvm.noduplicate() #2

attributes #0 = { norecurse nounwind }
attributes #1 = { norecurse nounwind readnone convergent }
attributes #2 = { norecurse nounwind readnone noduplicate }
