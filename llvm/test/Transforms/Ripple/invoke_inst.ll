; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce))' -S < %s | FileCheck %s --implicit-check-not="warning:"
; Function Attrs: mustprogress uwtable
define dso_local noundef float @_Z3fooi(i32 noundef %idx) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
entry:
  %call2.i.i.i.i3.i.i18 = tail call noalias noundef nonnull dereferenceable(128) ptr @_Znwm(i64 noundef 128) #6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(128) %call2.i.i.i.i3.i.i18, i8 0, i64 128, i1 false), !tbaa !5
  %call2.i.i.i.i3.i.i23 = invoke noalias noundef nonnull dereferenceable(128) ptr @_Znwm(i64 noundef 128) #6
          to label %_ZNSt6vectorIfSaIfEEC2EmRKS0_.exit24 unwind label %lpad2

_ZNSt6vectorIfSaIfEEC2EmRKS0_.exit24:             ; preds = %entry
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(128) %call2.i.i.i.i3.i.i23, i8 0, i64 128, i1 false), !tbaa !5
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %0 = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %conv4 = sitofp i32 %idx to float
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %conv9 = sext i32 %idx to i64
  %add.ptr.i = getelementptr inbounds float, ptr %call2.i.i.i.i3.i.i23, i64 %conv9
  %1 = load float, ptr %add.ptr.i, align 4, !tbaa !5
  tail call void @_ZdlPv(ptr noundef nonnull %call2.i.i.i.i3.i.i23) #7
  tail call void @_ZdlPv(ptr noundef nonnull %call2.i.i.i.i3.i.i18) #7
  ret float %1

lpad2:                                            ; preds = %entry
  %2 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZdlPv(ptr noundef nonnull %call2.i.i.i.i3.i.i18) #7
  resume { ptr, i32 } %2

for.body:                                         ; preds = %_ZNSt6vectorIfSaIfEEC2EmRKS0_.exit24, %for.body
  %indvars.iv = phi i64 [ 0, %_ZNSt6vectorIfSaIfEEC2EmRKS0_.exit24 ], [ %indvars.iv.next, %for.body ]
  %add = add i64 %0, %indvars.iv
  %add.ptr.i31 = getelementptr inbounds float, ptr %call2.i.i.i.i3.i.i18, i64 %add
  %3 = load float, ptr %add.ptr.i31, align 4, !tbaa !5
  %mul = fmul float %3, %conv4
; CHECK: fmul <8 x float>
  %add.ptr.i32 = getelementptr inbounds float, ptr %call2.i.i.i.i3.i.i23, i64 %add
  store float %mul, ptr %add.ptr.i32, align 4, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 8
  %cmp = icmp ult i64 %indvars.iv, 24
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !9
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #2

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #3

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(ptr noundef) local_unnamed_addr #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #5

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #3 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { allocsize(0) }
attributes #7 = { nounwind }

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
