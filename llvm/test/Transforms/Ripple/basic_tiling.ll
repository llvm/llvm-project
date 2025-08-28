; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce))' -S < %s | FileCheck %s --implicit-check-not="warning:"
source_filename = "another_shape_prop_bug.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable
define dso_local void @foo(ptr nocapture noundef readonly %A, ptr nocapture noundef writeonly %B) local_unnamed_addr #0 {
entry:
  %A_tile = alloca [10 x [10 x float]], align 16
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 10, i64 10, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  call void @llvm.lifetime.start.p0(i64 400, ptr nonnull %A_tile) #4
  %0 = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %1 = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 1)
  %arrayidx1 = getelementptr inbounds [10 x float], ptr %A, i64 %1, i64 %0
  ; CHECK: load <100 x float>
  %2 = load float, ptr %arrayidx1, align 4, !tbaa !5
  %arrayidx3 = getelementptr inbounds [10 x [10 x float]], ptr %A_tile, i64 0, i64 %1, i64 %0
  store float %2, ptr %arrayidx3, align 4, !tbaa !5
  %arrayidx5 = getelementptr inbounds [10 x [10 x float]], ptr %A_tile, i64 0, i64 %0, i64 %1
  ; CHECK: call <100 x float> @llvm.masked.gather.v100f32.v100p0
  %3 = load float, ptr %arrayidx5, align 4, !tbaa !5
  %arrayidx7 = getelementptr inbounds [10 x float], ptr %B, i64 %0, i64 %1
  ; CHECK: call void @llvm.masked.scatter.v100f32.v100p0
  store float %3, ptr %arrayidx7, align 4, !tbaa !5
  call void @llvm.lifetime.end.p0(i64 400, ptr nonnull %A_tile) #4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { nounwind }

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
!8 = !{!"Simple C/C++ TBAA"}
