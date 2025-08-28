; RUN: opt -passes=ripple -S %s | FileCheck %s --implicit-check-not="warning:"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; We check that the compilation was successful
; CHECK: @_ZN12_GLOBAL__N_111SoftmaxTestILj10ELj10EE3runEj
; CHECK-NOT: call{{.*}}@llvm.ripple.block.index

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable
define void @_ZN12_GLOBAL__N_111SoftmaxTestILj10ELj10EE3runEj(ptr noundef nonnull align 8 captures(none) dereferenceable(3856) %this, i32 %0) unnamed_addr #13 align 2 {
entry:
  %XRef = getelementptr inbounds nuw i8, ptr %this, i64 16
  %XSoftmax = getelementptr inbounds nuw i8, ptr %this, i64 2576
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 10, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %1 = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %add9.i = add i64 %1, 10
  %cmp10.i = icmp ugt i64 %1, -11
  br label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %if.end68.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %if.end68.i ]
  %arrayidx5.i = getelementptr inbounds nuw [10 x float], ptr %XRef, i64 %indvars.iv.i, i64 %1
  %2 = load float, ptr %arrayidx5.i, align 4, !tbaa !20
  br i1 %cmp10.i, label %if.then60.i, label %if.end44.i

if.end44.i:                                       ; preds = %for.cond1.preheader.i
  %3 = tail call noundef float @llvm.ripple.reduce.fmax.f32(i64 1, float %2)
  %cmp.i.i.i.i = fcmp ogt float %3, 0xC7EFFFFFE0000000
  %.sroa.speculated.i = select i1 %cmp.i.i.i.i, float %3, float 0xC7EFFFFFE0000000
  %sub.i = fsub float %2, %.sroa.speculated.i
  %4 = tail call float @llvm.exp.f32(float %sub.i)
  %5 = tail call noundef float @llvm.ripple.reduce.fadd.f32(i64 1, float %4)
  %add27.i = fadd float %5, 0.000000e+00
  %div.i = fdiv float %4, %add27.i
  br label %if.end68.i

if.then60.i:                                      ; preds = %for.cond1.preheader.i
  %arrayidx14.i = getelementptr inbounds nuw [10 x float], ptr %XRef, i64 %indvars.iv.i, i64 %add9.i
  %6 = load float, ptr %arrayidx14.i, align 4, !tbaa !20
  %7 = tail call noundef float @llvm.ripple.reduce.fmax.f32(i64 1, float %6)
  %sub.i4 = fsub float %2, %7
  %8 = tail call float @llvm.exp.f32(float %sub.i4)
  %9 = tail call noundef float @llvm.ripple.reduce.fadd.f32(i64 1, float %8)
  %add27.i5 = fadd float %9, 0.000000e+00
  %sub39.i = fsub float %6, %7
  %10 = tail call float @llvm.exp.f32(float %sub39.i)
  %11 = tail call noundef float @llvm.ripple.reduce.fadd.f32(i64 1, float %10)
  %add43.i = fadd float %11, %add27.i5
  %div127.i = fdiv float %8, %add43.i
  %arrayidx54128.i = getelementptr inbounds nuw [10 x float], ptr %XSoftmax, i64 %indvars.iv.i, i64 %1
  store float %div127.i, ptr %arrayidx54128.i, align 4, !tbaa !20
  %div63.i = fdiv float %10, %add43.i
  br label %if.end68.i

if.end68.i:                                       ; preds = %if.then60.i, %if.end44.i
  %.sink.i = phi i64 [ %1, %if.end44.i ], [ %add9.i, %if.then60.i ]
  %div.sink.i = phi float [ %div.i, %if.end44.i ], [ %div63.i, %if.then60.i ]
  %arrayidx54.i = getelementptr inbounds nuw [10 x float], ptr %XSoftmax, i64 %indvars.iv.i, i64 %.sink.i
  store float %div.sink.i, ptr %arrayidx54.i, align 4, !tbaa !20
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 32
  br i1 %exitcond.not.i, label %_ZN12_GLOBAL__N_17softmaxILj10ELj10EEEvjPAT__KfPAT__f.exit, label %for.cond1.preheader.i, !llvm.loop !22

_ZN12_GLOBAL__N_17softmaxILj10ELj10EEEvjPAT__KfPAT__f.exit: ; preds = %if.end68.i
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #3 = { mustprogress noreturn nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { inlinehint mustprogress noreturn nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #8 = { nofree nounwind }
attributes #9 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #12 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #13 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #14 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #15 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #16 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #17 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #18 = { mustprogress noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #19 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #20 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #21 = { noreturn }
attributes #22 = { noreturn nounwind }
attributes #23 = { builtin nounwind allocsize(0) }
attributes #24 = { builtin nounwind }
attributes #25 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR"}
!5 = !{!6, !7, i64 8}
!6 = !{!"_ZTSNSt3__16vectorINS_4pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEPN17ripple_test_suite4TestEEENS5_ISB_EEEE", !7, i64 0, !7, i64 8, !7, i64 16}
!7 = !{!"p1 _ZTSNSt3__14pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEPN17ripple_test_suite4TestEEE", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!6, !7, i64 16}
!12 = !{i64 0, i64 24, !13}
!13 = !{!9, !9, i64 0}
!14 = !{!15, !17, i64 24}
!15 = !{!"_ZTSNSt3__14pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEPN17ripple_test_suite4TestEEE", !16, i64 0, !17, i64 24}
!16 = !{!"_ZTSNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE", !9, i64 0}
!17 = !{!"p1 _ZTSN17ripple_test_suite4TestE", !8, i64 0}
!18 = !{!6, !7, i64 0}
!19 = !{!7, !7, i64 0}
!20 = !{!21, !21, i64 0}
!21 = !{!"float", !9, i64 0}
!22 = distinct !{!22, !23}
!23 = !{!"llvm.loop.mustprogress"}
!24 = distinct !{!24, !23}
!25 = !{!"branch_weights", !"expected", i32 1, i32 2000}
!26 = !{!"branch_weights", i32 1, i32 1048575}
!27 = !{!28, !28, i64 0}
!28 = !{!"p1 _ZTSN17ripple_test_suite13TestFrameworkE", !8, i64 0}
!29 = !{!30, !30, i64 0}
!30 = !{!"vtable pointer", !10, i64 0}
!31 = distinct !{!31, !23}
!32 = distinct !{!32, !23}
