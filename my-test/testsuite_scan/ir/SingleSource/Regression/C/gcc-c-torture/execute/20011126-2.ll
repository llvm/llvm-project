; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20011126-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20011126-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"aab\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [4 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #3
  br label %2

2:                                                ; preds = %44, %0
  %3 = phi ptr [ @.str, %0 ], [ %45, %44 ]
  %4 = phi ptr [ %1, %0 ], [ %46, %44 ]
  %5 = load i8, ptr %3, align 1, !tbaa !6
  %6 = icmp eq i8 %5, 97
  br label %7

7:                                                ; preds = %16, %2
  br i1 %6, label %8, label %16

8:                                                ; preds = %7, %8
  %9 = phi i64 [ %13, %8 ], [ 0, %7 ]
  %10 = phi ptr [ %11, %8 ], [ %3, %7 ]
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 1
  %12 = load i8, ptr %11, align 1, !tbaa !6
  %13 = add i64 %9, 1
  switch i8 %12, label %14 [
    i8 120, label %8
    i8 98, label %57
  ]

14:                                               ; preds = %8
  %15 = icmp ult ptr %3, %11
  br i1 %15, label %17, label %16

16:                                               ; preds = %14, %7
  br label %7

17:                                               ; preds = %14
  %18 = ptrtoint ptr %3 to i64
  %19 = ptrtoint ptr %4 to i64
  %20 = ptrtoint ptr %3 to i64
  %21 = add i64 %9, %18
  %22 = call i64 @llvm.usub.sat.i64(i64 %21, i64 %18)
  %23 = add i64 %22, 1
  %24 = icmp ult i64 %23, 32
  %25 = sub i64 %19, %20
  %26 = icmp ult i64 %25, 32
  %27 = select i1 %24, i1 true, i1 %26
  br i1 %27, label %47, label %28

28:                                               ; preds = %17
  %29 = and i64 %23, -32
  %30 = getelementptr i8, ptr %4, i64 %29
  %31 = getelementptr i8, ptr %3, i64 %29
  br label %32

32:                                               ; preds = %32, %28
  %33 = phi i64 [ 0, %28 ], [ %40, %32 ]
  %34 = getelementptr i8, ptr %4, i64 %33
  %35 = getelementptr i8, ptr %3, i64 %33
  %36 = getelementptr i8, ptr %35, i64 16
  %37 = load <16 x i8>, ptr %35, align 1, !tbaa !6
  %38 = load <16 x i8>, ptr %36, align 1, !tbaa !6
  %39 = getelementptr i8, ptr %34, i64 16
  store <16 x i8> %37, ptr %34, align 1, !tbaa !6
  store <16 x i8> %38, ptr %39, align 1, !tbaa !6
  %40 = add nuw i64 %33, 32
  %41 = icmp eq i64 %40, %29
  br i1 %41, label %42, label %32, !llvm.loop !9

42:                                               ; preds = %32
  %43 = icmp eq i64 %23, %29
  br i1 %43, label %44, label %47

44:                                               ; preds = %50, %42
  %45 = phi ptr [ %31, %42 ], [ %53, %50 ]
  %46 = phi ptr [ %30, %42 ], [ %55, %50 ]
  br label %2

47:                                               ; preds = %17, %42
  %48 = phi ptr [ %4, %17 ], [ %30, %42 ]
  %49 = phi ptr [ %3, %17 ], [ %31, %42 ]
  br label %50

50:                                               ; preds = %47, %50
  %51 = phi ptr [ %55, %50 ], [ %48, %47 ]
  %52 = phi ptr [ %53, %50 ], [ %49, %47 ]
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 1
  %54 = load i8, ptr %52, align 1, !tbaa !6
  %55 = getelementptr inbounds nuw i8, ptr %51, i64 1
  store i8 %54, ptr %51, align 1, !tbaa !6
  %56 = icmp ult ptr %52, %10
  br i1 %56, label %50, label %44, !llvm.loop !13

57:                                               ; preds = %8
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #3
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.usub.sat.i64(i64, i64) #2

attributes #0 = { nofree norecurse nosync nounwind memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !10, !11}
