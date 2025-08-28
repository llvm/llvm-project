; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce))' -S %s | FileCheck %s --implicit-check-not="warning:"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: write, inaccessiblemem: readwrite) uwtable
define dso_local void @_Z3foomPf(i64 noundef %N, ptr noundef writeonly captures(none) %a) local_unnamed_addr #0 {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %0 = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %cmp8 = icmp ult i64 %0, %N
  br i1 %cmp8, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %conv = uitofp i64 %0 to float
  %invariant.op = add i64 %0, 8
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %add10 = phi i64 [ %0, %for.body.lr.ph ], [ %add.reass, %for.body ]
  %i.09 = phi i64 [ 0, %for.body.lr.ph ], [ %add2, %for.body ]
  %arrayidx = getelementptr inbounds nuw float, ptr %a, i64 %add10
  store float %conv, ptr %arrayidx, align 4, !tbaa !5
  %add2 = add i64 %i.09, 8
  %add.reass = add nuw i64 %i.09, %invariant.op
  %cmp = icmp ult i64 %add.reass, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !9
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: write, inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
