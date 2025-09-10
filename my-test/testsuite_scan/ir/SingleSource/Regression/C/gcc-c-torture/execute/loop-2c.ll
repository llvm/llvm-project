; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/loop-2c.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/loop-2c.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local global [2 x i32] zeroinitializer, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @g(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %46, label %3

3:                                                ; preds = %1
  %4 = sext i32 %0 to i64
  %5 = getelementptr inbounds i32, ptr @a, i64 %4
  %6 = zext i32 %0 to i64
  %7 = icmp ult i32 %0, 8
  br i1 %7, label %35, label %8

8:                                                ; preds = %3
  %9 = and i64 %6, 4294967288
  %10 = mul nsw i64 %9, -4
  %11 = getelementptr i8, ptr %5, i64 %10
  %12 = trunc nuw i64 %9 to i32
  %13 = sub i32 %0, %12
  %14 = insertelement <4 x i32> poison, i32 %0, i64 0
  %15 = shufflevector <4 x i32> %14, <4 x i32> poison, <4 x i32> zeroinitializer
  %16 = add <4 x i32> %15, <i32 0, i32 -1, i32 -2, i32 -3>
  br label %17

17:                                               ; preds = %17, %8
  %18 = phi i64 [ 0, %8 ], [ %30, %17 ]
  %19 = phi <4 x i32> [ %16, %8 ], [ %31, %17 ]
  %20 = mul i64 %18, -4
  %21 = getelementptr i8, ptr %5, i64 %20
  %22 = mul <4 x i32> %19, splat (i32 3)
  %23 = mul <4 x i32> %19, splat (i32 3)
  %24 = add <4 x i32> %22, <i32 add (i32 ptrtoint (ptr @a to i32), i32 -3), i32 add (i32 ptrtoint (ptr @a to i32), i32 -3), i32 add (i32 ptrtoint (ptr @a to i32), i32 -3), i32 add (i32 ptrtoint (ptr @a to i32), i32 -3)>
  %25 = add <4 x i32> %23, <i32 add (i32 ptrtoint (ptr @a to i32), i32 -15), i32 add (i32 ptrtoint (ptr @a to i32), i32 -15), i32 add (i32 ptrtoint (ptr @a to i32), i32 -15), i32 add (i32 ptrtoint (ptr @a to i32), i32 -15)>
  %26 = getelementptr inbounds i8, ptr %21, i64 -16
  %27 = getelementptr inbounds i8, ptr %21, i64 -32
  %28 = shufflevector <4 x i32> %24, <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i32> %28, ptr %26, align 4, !tbaa !6
  %29 = shufflevector <4 x i32> %25, <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i32> %29, ptr %27, align 4, !tbaa !6
  %30 = add nuw i64 %18, 8
  %31 = add <4 x i32> %19, splat (i32 -8)
  %32 = icmp eq i64 %30, %9
  br i1 %32, label %33, label %17, !llvm.loop !10

33:                                               ; preds = %17
  %34 = icmp eq i64 %9, %6
  br i1 %34, label %46, label %35

35:                                               ; preds = %3, %33
  %36 = phi ptr [ %5, %3 ], [ %11, %33 ]
  %37 = phi i32 [ %0, %3 ], [ %13, %33 ]
  br label %38

38:                                               ; preds = %35, %38
  %39 = phi ptr [ %44, %38 ], [ %36, %35 ]
  %40 = phi i32 [ %41, %38 ], [ %37, %35 ]
  %41 = add i32 %40, -1
  %42 = mul i32 %41, 3
  %43 = add i32 %42, ptrtoint (ptr @a to i32)
  %44 = getelementptr inbounds i8, ptr %39, i64 -4
  store i32 %43, ptr %44, align 4, !tbaa !6
  %45 = icmp eq i32 %41, 0
  br i1 %45, label %46, label %38, !llvm.loop !14

46:                                               ; preds = %38, %33, %1
  ret void
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store i32 add (i32 ptrtoint (ptr @a to i32), i32 3), ptr getelementptr inbounds nuw (i8, ptr @a, i64 4), align 4, !tbaa !6
  store i32 ptrtoint (ptr @a to i32), ptr @a, align 4, !tbaa !6
  tail call void @exit(i32 noundef 0) #3
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !13, !12}
