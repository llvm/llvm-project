; ModuleID = 'massive_test.c'
source_filename = "massive_test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [12 x i8] c"Result: %d\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local i32 @massive_vreg_test(ptr noundef readonly captures(none) %input, ptr noundef writeonly captures(none) initializes((0, 40)) %output) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr %input, align 4, !tbaa !6
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %input, i64 4
  %1 = load i32, ptr %arrayidx1, align 4, !tbaa !6
  %add2 = add nsw i32 %1, 1
  %arrayidx197 = getelementptr inbounds nuw i8, ptr %input, i64 396
  %arrayidx395 = getelementptr inbounds nuw i8, ptr %input, i64 792
  %arrayidx593 = getelementptr inbounds nuw i8, ptr %input, i64 1188
  %arrayidx791 = getelementptr inbounds nuw i8, ptr %input, i64 1584
  %arrayidx989 = getelementptr inbounds nuw i8, ptr %input, i64 1980
  %arrayidx1187 = getelementptr inbounds nuw i8, ptr %input, i64 2376
  %arrayidx1385 = getelementptr inbounds nuw i8, ptr %input, i64 2772
  %arrayidx1583 = getelementptr inbounds nuw i8, ptr %input, i64 3168
  %arrayidx1781 = getelementptr inbounds nuw i8, ptr %input, i64 3564
  %2 = load i32, ptr %arrayidx1781, align 4, !tbaa !6
  %add1782 = add nsw i32 %2, 891
  %arrayidx1783 = getelementptr inbounds nuw i8, ptr %input, i64 3568
  %3 = load i32, ptr %arrayidx1783, align 4, !tbaa !6
  %add1784 = add nsw i32 %3, 892
  %mul = mul nsw i32 %add2, %0
  %mul2889 = mul nsw i32 %add1784, %add1782
  %arrayidx2998 = getelementptr inbounds nuw i8, ptr %output, i64 4
  %4 = load <2 x i32>, ptr %arrayidx197, align 4, !tbaa !6
  %5 = load <2 x i32>, ptr %arrayidx395, align 4, !tbaa !6
  %6 = shufflevector <2 x i32> %4, <2 x i32> %5, <2 x i32> <i32 0, i32 2>
  %7 = add nsw <2 x i32> %6, <i32 99, i32 198>
  %8 = shufflevector <2 x i32> %4, <2 x i32> %5, <2 x i32> <i32 1, i32 3>
  %9 = add nsw <2 x i32> %8, <i32 100, i32 199>
  %10 = mul nsw <2 x i32> %9, %7
  %arrayidx3000 = getelementptr inbounds nuw i8, ptr %output, i64 12
  %11 = load <2 x i32>, ptr %arrayidx593, align 4, !tbaa !6
  %12 = load <2 x i32>, ptr %arrayidx791, align 4, !tbaa !6
  %13 = shufflevector <2 x i32> %11, <2 x i32> %12, <2 x i32> <i32 0, i32 2>
  %14 = add nsw <2 x i32> %13, <i32 297, i32 396>
  %15 = shufflevector <2 x i32> %11, <2 x i32> %12, <2 x i32> <i32 1, i32 3>
  %16 = add nsw <2 x i32> %15, <i32 298, i32 397>
  %17 = mul nsw <2 x i32> %16, %14
  %arrayidx3002 = getelementptr inbounds nuw i8, ptr %output, i64 20
  %18 = load <2 x i32>, ptr %arrayidx989, align 4, !tbaa !6
  %19 = load <2 x i32>, ptr %arrayidx1187, align 4, !tbaa !6
  %20 = shufflevector <2 x i32> %18, <2 x i32> %19, <2 x i32> <i32 0, i32 2>
  %21 = add nsw <2 x i32> %20, <i32 495, i32 594>
  %22 = shufflevector <2 x i32> %18, <2 x i32> %19, <2 x i32> <i32 1, i32 3>
  %23 = add nsw <2 x i32> %22, <i32 496, i32 595>
  %24 = mul nsw <2 x i32> %23, %21
  %arrayidx3004 = getelementptr inbounds nuw i8, ptr %output, i64 28
  %25 = load <2 x i32>, ptr %arrayidx1385, align 4, !tbaa !6
  %26 = load <2 x i32>, ptr %arrayidx1583, align 4, !tbaa !6
  %27 = shufflevector <2 x i32> %25, <2 x i32> %26, <2 x i32> <i32 0, i32 2>
  %28 = add nsw <2 x i32> %27, <i32 693, i32 792>
  %29 = shufflevector <2 x i32> %25, <2 x i32> %26, <2 x i32> <i32 1, i32 3>
  %30 = add nsw <2 x i32> %29, <i32 694, i32 793>
  %31 = mul nsw <2 x i32> %30, %28
  store i32 %mul, ptr %output, align 4, !tbaa !6
  store <2 x i32> %10, ptr %arrayidx2998, align 4, !tbaa !6
  store <2 x i32> %17, ptr %arrayidx3000, align 4, !tbaa !6
  store <2 x i32> %24, ptr %arrayidx3002, align 4, !tbaa !6
  store <2 x i32> %31, ptr %arrayidx3004, align 4, !tbaa !6
  %arrayidx3006 = getelementptr inbounds nuw i8, ptr %output, i64 36
  store i32 %mul2889, ptr %arrayidx3006, align 4, !tbaa !6
  ret i32 %mul
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
entry:
  %input = alloca [1000 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %input) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %input, i8 0, i64 4000, i1 false)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %entry ], [ %vec.ind.next, %vector.body ]
  %step.add = add <4 x i32> %vec.ind, splat (i32 4)
  %0 = getelementptr inbounds nuw i32, ptr %input, i64 %index
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store <4 x i32> %vec.ind, ptr %0, align 4, !tbaa !6
  store <4 x i32> %step.add, ptr %1, align 4, !tbaa !6
  %index.next = add nuw i64 %index, 8
  %vec.ind.next = add <4 x i32> %vec.ind, splat (i32 8)
  %2 = icmp eq i64 %index.next, 1000
  br i1 %2, label %for.cond.cleanup, label %vector.body, !llvm.loop !10

for.cond.cleanup:                                 ; preds = %vector.body
  %3 = load i32, ptr %input, align 4, !tbaa !6
  %arrayidx1.i = getelementptr inbounds nuw i8, ptr %input, i64 4
  %4 = load i32, ptr %arrayidx1.i, align 4, !tbaa !6
  %add2.i = add nsw i32 %4, 1
  %mul.i = mul nsw i32 %add2.i, %3
  %call2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %mul.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %input) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git 9f790e9e900f8dab0e35b49a5844c2900865231e)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
