; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce))' -S < %s 2>&1 | FileCheck %s

; CHECK-NOT: warning
; CHECK-NOT: error

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local void @mandelbrot_ripple(float noundef %x0, float noundef %y0, float noundef %x1, float noundef %y1, i32 noundef %width, i32 noundef %height, i32 noundef %max_iters, ptr noundef writeonly captures(none) %output) local_unnamed_addr {
entry:
  %0 = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %sub = fsub float %x1, %x0
  %conv = sitofp i32 %width to float
  %div = fdiv float %sub, %conv
  %1 = tail call i64 @llvm.ripple.block.index.i64(ptr %0, i64 0)
  %conv1 = uitofp i64 %1 to float
  %2 = tail call float @llvm.fmuladd.f32(float %conv1, float %div, float %x0)
  %cmp.i9 = icmp sgt i32 %max_iters, 0
  br i1 %cmp.i9, label %for.body.i, label %mandelbrot_inner.exit

for.body.i:                                       ; preds = %entry, %for.inc.i
  %z_re.0.i12 = phi float [ %add.i, %for.inc.i ], [ %2, %entry ]
  %z_im.0.i11 = phi float [ %add5.i, %for.inc.i ], [ %y0, %entry ]
  %i.0.i10 = phi i32 [ %inc.i, %for.inc.i ], [ 0, %entry ]
  %mul1.i = fmul float %z_im.0.i11, %z_im.0.i11
  %3 = tail call float @llvm.fmuladd.f32(float %z_re.0.i12, float %z_re.0.i12, float %mul1.i)
  %cmp2.i = fcmp ogt float %3, 4.000000e+00
  br i1 %cmp2.i, label %mandelbrot_inner.exit, label %for.inc.i

for.inc.i:                                        ; preds = %for.body.i
  %neg.i = fneg float %mul1.i
  %4 = tail call float @llvm.fmuladd.f32(float %z_re.0.i12, float %z_re.0.i12, float %neg.i)
  %mul.i = fmul float %z_re.0.i12, 2.000000e+00
  %mul4.i = fmul float %z_im.0.i11, %mul.i
  %add.i = fadd float %2, %4
  %add5.i = fadd float %y0, %mul4.i
  %inc.i = add nuw nsw i32 %i.0.i10, 1
  %exitcond.not = icmp eq i32 %inc.i, %max_iters
  br i1 %exitcond.not, label %mandelbrot_inner.exit, label %for.body.i

mandelbrot_inner.exit:                            ; preds = %for.inc.i, %for.body.i, %entry
  %i.0.i.lcssa = phi i32 [ 0, %entry ], [ %i.0.i10, %for.body.i ], [ %max_iters, %for.inc.i ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %output, i64 %1
  store i32 %i.0.i.lcssa, ptr %arrayidx, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg)

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float)