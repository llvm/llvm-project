; RUN: llc --mtriple=loongarch64 -mattr=+d %s -o /dev/null

; ModuleID = 'bugpoint-reduced-simplifycfg.bc'
source_filename = "test-suite-src/MultiSource/Benchmarks/mafft/Lalignmm.c"

define float @Lalignmm_hmout(ptr %seq1, ptr %eff1, i32 %icyc) {
entry:
  %call4 = tail call i64 @strlen(ptr dereferenceable(1) poison)
  %conv5 = trunc i64 %call4 to i32
  %call7 = tail call i64 @strlen(ptr dereferenceable(1) poison)
  %call20 = tail call ptr @AllocateFloatVec(i32 signext poison)
  %call22 = tail call ptr @AllocateFloatVec(i32 signext poison)
  tail call void @st_OpeningGapCount(ptr poison, i32 signext %icyc, ptr %seq1, ptr %eff1, i32 signext %conv5)
  %sub110 = add nsw i32 %conv5, -1
  %sub111 = add nsw i32 0, -1
  br i1 poison, label %for.cond.preheader.i, label %if.end.i

for.cond.preheader.i:                             ; preds = %entry
  %sext294 = shl i64 %call4, 32
  %conv23.i = ashr exact i64 %sext294, 32
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.cond.preheader.i
  %call.i = tail call ptr @strncpy(ptr poison, ptr poison, i64 %conv23.i)
  br label %for.body.i

if.end.i:                                         ; preds = %entry
  %call82.i = tail call ptr @AllocateFloatVec(i32 signext poison)
  %call84.i = tail call ptr @AllocateFloatVec(i32 signext poison)
  %call86.i = tail call ptr @AllocateFloatVec(i32 signext poison)
  %call88.i = tail call ptr @AllocateFloatVec(i32 signext poison)
  %call90.i = tail call ptr @AllocateFloatVec(i32 signext poison)
  %call92.i = tail call ptr @AllocateIntVec(i32 signext poison)
  %call94.i = tail call ptr @AllocateIntVec(i32 signext poison)
  %call104.i = tail call ptr @AllocateFloatVec(i32 signext poison)
  %call108.i = tail call ptr @AllocateFloatVec(i32 signext poison)
  %call110.i = tail call ptr @AllocateIntVec(i32 signext poison)
  %idxprom220.i = sext i32 %sub111 to i64
  %mpjpt.018.i = getelementptr inbounds i32, ptr %call110.i, i64 1
  %arrayidx329.i = getelementptr inbounds float, ptr %call108.i, i64 %idxprom220.i
  %idxprom332.i = and i64 %call7, 4294967295
  %wide.trip.count130.i = zext i32 poison to i64
  %0 = add nsw i64 1, -1
  %arrayidx239.i = getelementptr inbounds float, ptr %call104.i, i64 1
  %1 = load float, ptr %arrayidx239.i, align 4
  store float %1, ptr %call84.i, align 4
  %curpt.017.i = getelementptr inbounds float, ptr %call84.i, i64 1
  %arrayidx279.i = getelementptr inbounds float, ptr %call20, i64 %0
  %2 = load ptr, ptr poison, align 8
  %3 = load ptr, ptr null, align 8
  %4 = trunc i64 %0 to i32
  br label %for.body260.us.i

for.body260.us.i:                                 ; preds = %if.end292.us.i, %if.end.i
  %indvars.iv132.i = phi i64 [ %indvars.iv.next133.i, %if.end292.us.i ], [ 1, %if.end.i ]
  %mpjpt.026.us.i = phi ptr [ poison, %if.end292.us.i ], [ %mpjpt.018.i, %if.end.i ]
  %curpt.025.us.i = phi ptr [ %curpt.0.us.i, %if.end292.us.i ], [ %curpt.017.i, %if.end.i ]
  %prept.022.us.i = phi ptr [ %incdec.ptr316.us.i, %if.end292.us.i ], [ %call82.i, %if.end.i ]
  %mi.021.us.i = phi float [ %mi.1.us.i, %if.end292.us.i ], [ poison, %if.end.i ]
  %5 = load float, ptr %prept.022.us.i, align 4
  %6 = add nsw i64 %indvars.iv132.i, -1
  %arrayidx263.us.i = getelementptr inbounds float, ptr %call22, i64 %6
  %7 = load float, ptr %arrayidx263.us.i, align 4
  %add264.us.i = fadd float %mi.021.us.i, %7
  %cmp265.us.i = fcmp ogt float %add264.us.i, %5
  %wm.0.us.i = select i1 %cmp265.us.i, float %add264.us.i, float %5
  %arrayidx270.us.i = getelementptr inbounds float, ptr poison, i64 %indvars.iv132.i
  %cmp272.us.i = fcmp ult float 0.000000e+00, %mi.021.us.i
  %mi.1.us.i = select i1 %cmp272.us.i, float %mi.021.us.i, float 0.000000e+00
  %8 = trunc i64 %6 to i32
  %mpi.1.us.i = select i1 %cmp272.us.i, i32 0, i32 %8
  %9 = load float, ptr %arrayidx279.i, align 4
  %add280.us.i = fadd float 0.000000e+00, %9
  %cmp281.us.i = fcmp ogt float %add280.us.i, %wm.0.us.i
  %wm.1.us.i = select i1 %cmp281.us.i, float %add280.us.i, float %wm.0.us.i
  %cmp288.us.i = fcmp ult float poison, 0.000000e+00
  br i1 %cmp288.us.i, label %if.end292.us.i, label %if.then290.us.i

if.then290.us.i:                                  ; preds = %for.body260.us.i
  store i32 %4, ptr %mpjpt.026.us.i, align 4
  br label %if.end292.us.i

if.end292.us.i:                                   ; preds = %if.then290.us.i, %for.body260.us.i
  %10 = phi i32 [ %4, %if.then290.us.i ], [ poison, %for.body260.us.i ]
  %add293.us.i = fadd float %wm.1.us.i, 0.000000e+00
  %arrayidx297.us.i = getelementptr inbounds float, ptr %2, i64 %indvars.iv132.i
  store float %add293.us.i, ptr %arrayidx297.us.i, align 4
  %arrayidx306.us.i = getelementptr inbounds i32, ptr %call94.i, i64 %indvars.iv132.i
  store i32 %10, ptr %arrayidx306.us.i, align 4
  %arrayidx308.us.i = getelementptr inbounds i32, ptr %call92.i, i64 %indvars.iv132.i
  store i32 %mpi.1.us.i, ptr %arrayidx308.us.i, align 4
  %11 = load float, ptr %curpt.025.us.i, align 4
  %arrayidx310.us.i = getelementptr inbounds float, ptr %call86.i, i64 %indvars.iv132.i
  store float %11, ptr %arrayidx310.us.i, align 4
  %arrayidx312.us.i = getelementptr inbounds float, ptr %call90.i, i64 %indvars.iv132.i
  store float 0.000000e+00, ptr %arrayidx312.us.i, align 4
  %arrayidx314.us.i = getelementptr inbounds float, ptr %call88.i, i64 %indvars.iv132.i
  store float %mi.1.us.i, ptr %arrayidx314.us.i, align 4
  %incdec.ptr316.us.i = getelementptr inbounds float, ptr %prept.022.us.i, i64 1
  %indvars.iv.next133.i = add nuw nsw i64 %indvars.iv132.i, 1
  %curpt.0.us.i = getelementptr inbounds float, ptr %curpt.025.us.i, i64 1
  %exitcond137.not.i = icmp eq i64 %indvars.iv.next133.i, %wide.trip.count130.i
  br i1 %exitcond137.not.i, label %for.end321.i, label %for.body260.us.i

for.end321.i:                                     ; preds = %if.end292.us.i
  %12 = load float, ptr %arrayidx329.i, align 4
  %arrayidx333.i = getelementptr inbounds float, ptr %3, i64 %idxprom332.i
  store float %12, ptr %arrayidx333.i, align 4
  tail call fastcc void @match_calc(ptr %call104.i, ptr poison, ptr poison, i32 signext %sub111, i32 signext %conv5, ptr poison, ptr poison, i32 signext 1)
  br label %for.body429.i

for.body429.i:                                    ; preds = %for.body429.i, %for.end321.i
  %j.743.i = phi i32 [ %sub111, %for.end321.i ], [ %sub436.i, %for.body429.i ]
  %sub436.i = add nsw i32 %j.743.i, -1
  %idxprom437.i = zext i32 %sub436.i to i64
  %arrayidx438.i = getelementptr inbounds float, ptr %call108.i, i64 %idxprom437.i
  store float 0.000000e+00, ptr %arrayidx438.i, align 4
  store i32 %sub110, ptr poison, align 4
  br label %for.body429.i
}

declare i64 @strlen(ptr)
declare ptr @AllocateFloatVec(i32)
declare void @st_OpeningGapCount(ptr, i32, ptr, ptr, i32)
declare ptr @strncpy(ptr, ptr, i64)
declare ptr @AllocateIntVec(i32)
declare void @match_calc(ptr, ptr, ptr, i32, i32, ptr, ptr, i32)
