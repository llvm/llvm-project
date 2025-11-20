; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #0
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0
declare signext i16 @cat(i16 signext) #1
declare void @danny(i16 signext, i16 signext, i16 signext, ptr nocapture readonly, i16 signext, ptr nocapture) #1
declare void @sammy(ptr nocapture readonly, ptr nocapture readonly, ptr nocapture readonly, ptr nocapture, ptr nocapture, i16 signext, i16 signext, i16 signext) #1
declare ptr @llvm.stacksave() #2
declare void @llvm.stackrestore(ptr) #2

define i32 @fred(i16 signext %p0, i16 signext %p1, ptr nocapture readonly %p2, i16 signext %p3, ptr nocapture readonly %p4, ptr nocapture %p5) #1 {
entry:
  %0 = tail call ptr @llvm.stacksave()
  %vla = alloca i16, i32 undef, align 8
  %call17 = call signext i16 @cat(i16 signext 1) #1
  br i1 undef, label %for.cond23.preheader, label %for.end47

for.cond23.preheader:                             ; preds = %for.end40, %entry
  %i.190 = phi i16 [ %inc46, %for.end40 ], [ 0, %entry ]
  br i1 undef, label %for.body27, label %for.end40

for.body27:                                       ; preds = %for.body27, %for.cond23.preheader
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body27 ], [ 0, %for.cond23.preheader ]
  %call30 = call signext i16 @cat(i16 signext 7) #1
  %arrayidx32 = getelementptr inbounds i16, ptr %vla, i32 %indvars.iv
  store i16 %call30, ptr %arrayidx32, align 2
  %arrayidx37 = getelementptr inbounds i16, ptr undef, i32 %indvars.iv
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i16 undef, %p3
  br i1 %exitcond, label %for.end40, label %for.body27

for.end40:                                        ; preds = %for.body27, %for.cond23.preheader
  call void @sammy(ptr nonnull undef, ptr undef, ptr %p4, ptr null, ptr undef, i16 signext undef, i16 signext undef, i16 signext undef) #1
  %inc46 = add nuw nsw i16 %i.190, 1
  %exitcond94 = icmp eq i16 %inc46, %call17
  br i1 %exitcond94, label %for.end47.loopexit, label %for.cond23.preheader

for.end47.loopexit:                               ; preds = %for.end40
  %.pre = load i16, ptr undef, align 2
  br label %for.end47

for.end47:                                        ; preds = %for.end47.loopexit, %entry
  %1 = phi i16 [ %.pre, %for.end47.loopexit ], [ 0, %entry ]
  call void @danny(i16 signext %1, i16 signext %p0, i16 signext %p1, ptr %p2, i16 signext %p3, ptr %p5) #1
  call void @llvm.stackrestore(ptr %0)
  ret i32 undef
}


attributes #0 = { argmemonly nounwind }
attributes #1 = { optsize }
attributes #2 = { nounwind }
