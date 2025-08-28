; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce,instcombine))' -S %s | FileCheck %s --implicit-check-not="warning:"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable
define dso_local void @foo(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, ptr nocapture noundef writeonly %apb, ptr nocapture noundef readonly %c, ptr nocapture noundef readonly %d, ptr nocapture noundef writeonly %cpd) local_unnamed_addr #0 {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %0 = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %1 = load <8 x float>, ptr %a, align 32, !tbaa !5
  %2 = load <8 x float>, ptr %b, align 32, !tbaa !5
  %add = fadd <8 x float> %1, %2
  store <8 x float> %add, ptr %apb, align 32, !tbaa !5
  %arrayidx = getelementptr inbounds float, ptr %c, i64 %0
  %3 = load float, ptr %arrayidx, align 4, !tbaa !8
  %arrayidx1 = getelementptr inbounds float, ptr %d, i64 %0
  %4 = load float, ptr %arrayidx1, align 4, !tbaa !8
  %add2 = fadd float %3, %4
  %arrayidx3 = getelementptr inbounds float, ptr %cpd, i64 %0
  store float %add2, ptr %arrayidx3, align 4, !tbaa !8
  ret void

; CHECK-LABEL: foo
; CHECK: load <8 x float>, ptr %a
; CHECK: load <8 x float>, ptr %b
; CHECK: [[APB:%.*]] = fadd <8 x float>
; CHECK: store <8 x float> [[APB]], ptr %apb
; CHECK: load <8 x float>, ptr %c
; CHECK: load <8 x float>, ptr %d
; CHECK: [[CPD:%.*]] = fadd <8 x float>
; CHECK: store <8 x float> [[CPD]], ptr %cpd
; CHECK: ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !6, i64 0}
