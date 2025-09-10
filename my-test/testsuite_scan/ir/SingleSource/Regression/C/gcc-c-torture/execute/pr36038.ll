; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr36038.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr36038.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@expect = dso_local global [10 x i64] [i64 0, i64 1, i64 2, i64 3, i64 4, i64 4, i64 5, i64 6, i64 7, i64 9], align 8
@stack_base = dso_local local_unnamed_addr global ptr null, align 8
@markstack_ptr = dso_local local_unnamed_addr global ptr null, align 8
@list = dso_local global [10 x i64] zeroinitializer, align 16
@indices = dso_local global [10 x i32] zeroinitializer, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @doit() local_unnamed_addr #0 {
  %1 = load ptr, ptr @markstack_ptr, align 8, !tbaa !6
  %2 = getelementptr inbounds i8, ptr %1, i64 -4
  %3 = load i32, ptr %2, align 4, !tbaa !11
  %4 = icmp eq i32 %3, 6
  br i1 %4, label %59, label %5

5:                                                ; preds = %0
  %6 = sub i32 6, %3
  %7 = load ptr, ptr @stack_base, align 8, !tbaa !13
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 40
  %9 = getelementptr inbounds i8, ptr %1, i64 -8
  %10 = load i32, ptr %9, align 4, !tbaa !11
  %11 = sub i32 %10, %3
  %12 = add i32 %11, 2
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds i64, ptr %8, i64 %13
  %15 = sub i32 5, %3
  %16 = zext i32 %15 to i64
  %17 = add nuw nsw i64 %16, 1
  %18 = icmp ult i32 %15, 3
  %19 = mul nsw i64 %13, -8
  %20 = icmp ult i64 %19, 32
  %21 = select i1 %18, i1 true, i1 %20
  br i1 %21, label %46, label %22

22:                                               ; preds = %5
  %23 = and i64 %17, 8589934588
  %24 = trunc i64 %23 to i32
  %25 = sub i32 %6, %24
  %26 = mul nsw i64 %23, -8
  %27 = getelementptr i8, ptr %8, i64 %26
  %28 = mul nsw i64 %23, -8
  %29 = getelementptr i8, ptr %14, i64 %28
  br label %30

30:                                               ; preds = %30, %22
  %31 = phi i64 [ 0, %22 ], [ %42, %30 ]
  %32 = mul i64 %31, -8
  %33 = getelementptr i8, ptr %8, i64 %32
  %34 = mul i64 %31, -8
  %35 = getelementptr i8, ptr %14, i64 %34
  %36 = getelementptr i8, ptr %33, i64 -8
  %37 = getelementptr i8, ptr %33, i64 -24
  %38 = load <2 x i64>, ptr %36, align 8, !tbaa !15
  %39 = load <2 x i64>, ptr %37, align 8, !tbaa !15
  %40 = getelementptr i8, ptr %35, i64 -8
  %41 = getelementptr i8, ptr %35, i64 -24
  store <2 x i64> %38, ptr %40, align 8, !tbaa !15
  store <2 x i64> %39, ptr %41, align 8, !tbaa !15
  %42 = add nuw i64 %31, 4
  %43 = icmp eq i64 %42, %23
  br i1 %43, label %44, label %30, !llvm.loop !17

44:                                               ; preds = %30
  %45 = icmp eq i64 %17, %23
  br i1 %45, label %59, label %46

46:                                               ; preds = %5, %44
  %47 = phi i32 [ %6, %5 ], [ %25, %44 ]
  %48 = phi ptr [ %8, %5 ], [ %27, %44 ]
  %49 = phi ptr [ %14, %5 ], [ %29, %44 ]
  br label %50

50:                                               ; preds = %46, %50
  %51 = phi i32 [ %57, %50 ], [ %47, %46 ]
  %52 = phi ptr [ %54, %50 ], [ %48, %46 ]
  %53 = phi ptr [ %56, %50 ], [ %49, %46 ]
  %54 = getelementptr inbounds i8, ptr %52, i64 -8
  %55 = load i64, ptr %52, align 8, !tbaa !15
  %56 = getelementptr inbounds i8, ptr %53, i64 -8
  store i64 %55, ptr %53, align 8, !tbaa !15
  %57 = add nsw i32 %51, -1
  %58 = icmp eq i32 %57, 0
  br i1 %58, label %59, label %50, !llvm.loop !21

59:                                               ; preds = %50, %44, %0
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store <2 x i64> <i64 0, i64 1>, ptr @list, align 16, !tbaa !15
  store <2 x i64> <i64 2, i64 3>, ptr getelementptr inbounds nuw (i8, ptr @list, i64 16), align 16, !tbaa !15
  store <2 x i64> <i64 4, i64 5>, ptr getelementptr inbounds nuw (i8, ptr @list, i64 32), align 16, !tbaa !15
  store <2 x i64> <i64 6, i64 7>, ptr getelementptr inbounds nuw (i8, ptr @list, i64 48), align 16, !tbaa !15
  store i64 9, ptr getelementptr inbounds nuw (i8, ptr @list, i64 72), align 8, !tbaa !15
  store ptr getelementptr inbounds nuw (i8, ptr @indices, i64 36), ptr @markstack_ptr, align 8, !tbaa !6
  store <2 x i32> <i32 1, i32 2>, ptr getelementptr inbounds nuw (i8, ptr @indices, i64 28), align 4, !tbaa !11
  store ptr getelementptr inbounds nuw (i8, ptr @list, i64 16), ptr @stack_base, align 8, !tbaa !13
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) getelementptr inbounds nuw (i8, ptr @list, i64 40), ptr noundef nonnull align 8 dereferenceable(32) getelementptr inbounds nuw (i8, ptr @list, i64 32), i64 32, i1 false), !tbaa !15
  %1 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(80) @expect, ptr noundef nonnull dereferenceable(80) @list, i64 80)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

4:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #4

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 int", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 long long", !8, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"long long", !9, i64 0}
!17 = distinct !{!17, !18, !19, !20}
!18 = !{!"llvm.loop.mustprogress"}
!19 = !{!"llvm.loop.isvectorized", i32 1}
!20 = !{!"llvm.loop.unroll.runtime.disable"}
!21 = distinct !{!21, !18, !19}
