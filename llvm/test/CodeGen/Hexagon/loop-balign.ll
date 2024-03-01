; RUN: llc -march=hexagon -O3  < %s | FileCheck %s -check-prefix=BALIGN
; BALIGN: .p2align{{.*}}5

; The test for checking the alignment of 'for.body4.for.body4_crit_edge' basic block

define dso_local void @foo(i32 %nCol, i32 %nRow, ptr nocapture %resMat) local_unnamed_addr {
entry:
  %shl = shl i32 %nRow, 2
  %cmp36 = icmp sgt i32 %nRow, 0
  %0 = add i32 %nCol, -1
  %.inv = icmp slt i32 %0, 1
  %1 = select i1 %.inv, i32 1, i32 %nCol
  br label %Outerloop

Outerloop:                                        ; preds = %for.end7, %entry
  %r12.0 = phi i32 [ 0, %entry ], [ %inc8, %for.end7 ]
  %r7_6.0 = phi i64 [ undef, %entry ], [ %r7_6.1.lcssa, %for.end7 ]
  %r0i.0 = phi i32 [ undef, %entry ], [ %r0i.1.lcssa, %for.end7 ]
  %r5.0 = phi ptr [ %resMat, %entry ], [ %r5.1.lcssa, %for.end7 ]
  %r8.0 = phi i32 [ %shl, %entry ], [ %r8.1.lcssa, %for.end7 ]
  br i1 %cmp36, label %for.body.lr.ph, label %for.end7

for.body.lr.ph:                                   ; preds = %Outerloop
  %cmp332 = icmp eq i32 %r12.0, 0
  %exitcond.peel = icmp eq i32 %r12.0, 1
  br label %for.body

for.body:                                         ; preds = %for.end, %for.body.lr.ph
  %r8.141 = phi i32 [ %r8.0, %for.body.lr.ph ], [ %add, %for.end ]
  %r5.140 = phi ptr [ %r5.0, %for.body.lr.ph ], [ %add.ptr, %for.end ]
  %i.039 = phi i32 [ 0, %for.body.lr.ph ], [ %inc6, %for.end ]
  %r0i.138 = phi i32 [ %r0i.0, %for.body.lr.ph ], [ %4, %for.end ]
  %r7_6.137 = phi i64 [ %r7_6.0, %for.body.lr.ph ], [ %r7_6.2.lcssa, %for.end ]
  %add = add nsw i32 %r8.141, %shl
  br i1 %cmp332, label %for.end, label %for.body4.peel

for.body4.peel:                                   ; preds = %for.body
  %r1i.0.in.peel = inttoptr i32 %r8.141 to ptr
  %r1i.0.peel = load i32, ptr %r1i.0.in.peel, align 4
  %2 = tail call i64 @llvm.hexagon.M2.dpmpyss.nac.s0(i64 %r7_6.137, i32 %r1i.0.peel, i32 %r0i.138)
  br i1 %exitcond.peel, label %for.end, label %for.body4.preheader.peel.newph

for.body4.preheader.peel.newph:                   ; preds = %for.body4.peel
  %r1i.0.in = inttoptr i32 %add to ptr
  %r1i.0 = load i32, ptr %r1i.0.in, align 4
  br label %for.body4

for.body4:                                        ; preds = %for.body4.for.body4_crit_edge, %for.body4.preheader.peel.newph
  %inc.phi = phi i32 [ %inc.0, %for.body4.for.body4_crit_edge ], [ 2, %for.body4.preheader.peel.newph ]
  %r7_6.233 = phi i64 [ %3, %for.body4.for.body4_crit_edge ], [ %2, %for.body4.preheader.peel.newph ]
  %3 = tail call i64 @llvm.hexagon.M2.dpmpyss.nac.s0(i64 %r7_6.233, i32 %r1i.0, i32 %r0i.138)
  %exitcond = icmp eq i32 %inc.phi, %r12.0
  br i1 %exitcond, label %for.end.loopexit, label %for.body4.for.body4_crit_edge

for.body4.for.body4_crit_edge:                    ; preds = %for.body4
  %inc.0 = add nuw nsw i32 %inc.phi, 1
  br label %for.body4

for.end.loopexit:                                 ; preds = %for.body4
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.body4.peel, %for.body
  %r7_6.2.lcssa = phi i64 [ %r7_6.137, %for.body ], [ %2, %for.body4.peel ], [ %3, %for.end.loopexit ]
  %4 = tail call i32 @llvm.hexagon.S2.clbp(i64 %r7_6.2.lcssa)
  store i32 %4, ptr %r5.140, align 4
  %add.ptr = getelementptr inbounds i8, ptr %r5.140, i32 undef
  %inc6 = add nuw nsw i32 %i.039, 1
  %exitcond47 = icmp eq i32 %inc6, %nRow
  br i1 %exitcond47, label %for.end7.loopexit, label %for.body

for.end7.loopexit:                                ; preds = %for.end
  br label %for.end7

for.end7:                                         ; preds = %for.end7.loopexit, %Outerloop
  %r7_6.1.lcssa = phi i64 [ %r7_6.0, %Outerloop ], [ %r7_6.2.lcssa, %for.end7.loopexit ]
  %r0i.1.lcssa = phi i32 [ %r0i.0, %Outerloop ], [ %4, %for.end7.loopexit ]
  %r5.1.lcssa = phi ptr [ %r5.0, %Outerloop ], [ %add.ptr, %for.end7.loopexit ]
  %r8.1.lcssa = phi i32 [ %r8.0, %Outerloop ], [ %add, %for.end7.loopexit ]
  %inc8 = add nuw i32 %r12.0, 1
  %exitcond48 = icmp eq i32 %inc8, %1
  br i1 %exitcond48, label %if.end, label %Outerloop

if.end:                                           ; preds = %for.end7
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.dpmpyss.nac.s0(i64, i32, i32) 

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.clbp(i64)
