; RUN: llc -mtriple=hexagon -O2 < %s
; Check for successful compilation. It originally caused an abort due to
; the "isBarrier" flag set on instructions that were not meant to have it.

target triple = "hexagon"

; Function Attrs: nounwind optsize readnone
define void @dummy() #0 {
entry:
  ret void
}

; Function Attrs: nounwind optsize
define void @conv3x3(ptr nocapture readonly %inp, ptr nocapture readonly %mask, i32 %shift, ptr nocapture %outp, i32 %width) #1 {
entry:
  %cmp381 = icmp sgt i32 %width, 0
  %arrayidx16.gep = getelementptr i8, ptr %mask, i32 4
  %arrayidx19.gep = getelementptr i8, ptr %mask, i32 8
  br label %for.body

for.body:                                         ; preds = %for.inc48, %entry
  %i.086 = phi i32 [ 0, %entry ], [ %inc49, %for.inc48 ]
  %mul = mul nsw i32 %i.086, %width
  %arrayidx.sum = add i32 %mul, %width
  br i1 %cmp381, label %for.cond5.preheader.lr.ph, label %for.inc48

for.cond5.preheader.lr.ph:                        ; preds = %for.body
  %add.ptr.sum = add i32 %arrayidx.sum, %width
  %add.ptr1 = getelementptr inbounds i8, ptr %inp, i32 %add.ptr.sum
  %add.ptr = getelementptr inbounds i8, ptr %inp, i32 %arrayidx.sum
  %arrayidx = getelementptr inbounds i8, ptr %inp, i32 %mul
  %arrayidx44.gep = getelementptr i8, ptr %outp, i32 %mul
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %if.end40, %for.cond5.preheader.lr.ph
  %arrayidx44.phi = phi ptr [ %arrayidx44.gep, %for.cond5.preheader.lr.ph ], [ %arrayidx44.inc, %if.end40 ]
  %j.085 = phi i32 [ 0, %for.cond5.preheader.lr.ph ], [ %inc46, %if.end40 ]
  %IN1.084 = phi ptr [ %arrayidx, %for.cond5.preheader.lr.ph ], [ %incdec.ptr, %if.end40 ]
  %IN2.083 = phi ptr [ %add.ptr, %for.cond5.preheader.lr.ph ], [ %incdec.ptr33, %if.end40 ]
  %IN3.082 = phi ptr [ %add.ptr1, %for.cond5.preheader.lr.ph ], [ %incdec.ptr34, %if.end40 ]
  br label %for.body7

for.body7:                                        ; preds = %for.body7, %for.cond5.preheader
  %arrayidx8.phi = phi ptr [ %IN1.084, %for.cond5.preheader ], [ %arrayidx8.inc, %for.body7 ]
  %arrayidx9.phi = phi ptr [ %IN2.083, %for.cond5.preheader ], [ %arrayidx9.inc, %for.body7 ]
  %arrayidx11.phi = phi ptr [ %IN3.082, %for.cond5.preheader ], [ %arrayidx11.inc, %for.body7 ]
  %arrayidx13.phi = phi ptr [ %mask, %for.cond5.preheader ], [ %arrayidx13.inc, %for.body7 ]
  %arrayidx16.phi = phi ptr [ %arrayidx16.gep, %for.cond5.preheader ], [ %arrayidx16.inc, %for.body7 ]
  %arrayidx19.phi = phi ptr [ %arrayidx19.gep, %for.cond5.preheader ], [ %arrayidx19.inc, %for.body7 ]
  %k.080 = phi i32 [ 0, %for.cond5.preheader ], [ %inc, %for.body7 ]
  %sum.079 = phi i32 [ 0, %for.cond5.preheader ], [ %add32, %for.body7 ]
  %0 = load i8, ptr %arrayidx8.phi, align 1, !tbaa !1
  %1 = load i8, ptr %arrayidx9.phi, align 1, !tbaa !1
  %2 = load i8, ptr %arrayidx11.phi, align 1, !tbaa !1
  %3 = load i8, ptr %arrayidx13.phi, align 1, !tbaa !1
  %4 = load i8, ptr %arrayidx16.phi, align 1, !tbaa !1
  %5 = load i8, ptr %arrayidx19.phi, align 1, !tbaa !1
  %conv21 = zext i8 %0 to i32
  %conv22 = sext i8 %3 to i32
  %mul23 = mul nsw i32 %conv22, %conv21
  %conv24 = zext i8 %1 to i32
  %conv25 = sext i8 %4 to i32
  %mul26 = mul nsw i32 %conv25, %conv24
  %conv27 = zext i8 %2 to i32
  %conv28 = sext i8 %5 to i32
  %mul29 = mul nsw i32 %conv28, %conv27
  %add30 = add i32 %mul23, %sum.079
  %add31 = add i32 %add30, %mul26
  %add32 = add i32 %add31, %mul29
  %inc = add nsw i32 %k.080, 1
  %exitcond = icmp eq i32 %inc, 3
  %arrayidx8.inc = getelementptr i8, ptr %arrayidx8.phi, i32 1
  %arrayidx9.inc = getelementptr i8, ptr %arrayidx9.phi, i32 1
  %arrayidx11.inc = getelementptr i8, ptr %arrayidx11.phi, i32 1
  %arrayidx13.inc = getelementptr i8, ptr %arrayidx13.phi, i32 1
  %arrayidx16.inc = getelementptr i8, ptr %arrayidx16.phi, i32 1
  %arrayidx19.inc = getelementptr i8, ptr %arrayidx19.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body7

for.end:                                          ; preds = %for.body7
  %incdec.ptr = getelementptr inbounds i8, ptr %IN1.084, i32 1
  %incdec.ptr33 = getelementptr inbounds i8, ptr %IN2.083, i32 1
  %incdec.ptr34 = getelementptr inbounds i8, ptr %IN3.082, i32 1
  %shr = ashr i32 %add32, %shift
  %cmp35 = icmp slt i32 %shr, 0
  br i1 %cmp35, label %if.end40, label %if.end

if.end:                                           ; preds = %for.end
  %cmp37 = icmp sgt i32 %shr, 255
  br i1 %cmp37, label %if.then39, label %if.end40

if.then39:                                        ; preds = %if.end
  br label %if.end40

if.end40:                                         ; preds = %for.end, %if.then39, %if.end
  %sum.2 = phi i32 [ 255, %if.then39 ], [ %shr, %if.end ], [ 0, %for.end ]
  %conv41 = trunc i32 %sum.2 to i8
  store i8 %conv41, ptr %arrayidx44.phi, align 1, !tbaa !1
  %inc46 = add nsw i32 %j.085, 1
  %exitcond87 = icmp eq i32 %inc46, %width
  %arrayidx44.inc = getelementptr i8, ptr %arrayidx44.phi, i32 1
  br i1 %exitcond87, label %for.inc48.loopexit, label %for.cond5.preheader

for.inc48.loopexit:                               ; preds = %if.end40
  br label %for.inc48

for.inc48:                                        ; preds = %for.inc48.loopexit, %for.body
  %inc49 = add nsw i32 %i.086, 1
  %exitcond88 = icmp eq i32 %inc49, 2
  br i1 %exitcond88, label %for.end50, label %for.body

for.end50:                                        ; preds = %for.inc48
  ret void
}

attributes #0 = { nounwind optsize readnone }
attributes #1 = { nounwind optsize }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
