; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20030717-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20030717-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.B = type { i32, i32, i32, i32, i32 }

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local i32 @bar(ptr noundef captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %4 = load i32, ptr %3, align 4, !tbaa !6
  %5 = load i16, ptr %1, align 8, !tbaa !11
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load i64, ptr %6, align 8, !tbaa !15
  %8 = sext i32 %4 to i64
  %9 = getelementptr inbounds %struct.B, ptr %0, i64 %8, i32 3
  %10 = load i32, ptr %9, align 4, !tbaa !16
  %11 = trunc i64 %7 to i32
  %12 = sub i32 %11, %10
  %13 = tail call range(i32 0, -2147483648) i32 @llvm.abs.i32(i32 %12, i1 true)
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 20
  br label %15

15:                                               ; preds = %21, %2
  %16 = phi i32 [ %4, %2 ], [ %23, %21 ]
  %17 = phi i32 [ %4, %2 ], [ %30, %21 ]
  %18 = icmp slt i32 %16, 1
  br i1 %18, label %19, label %21

19:                                               ; preds = %15
  %20 = load i32, ptr %14, align 4, !tbaa !18
  br label %21

21:                                               ; preds = %19, %15
  %22 = phi i32 [ %20, %19 ], [ %16, %15 ]
  %23 = add nsw i32 %22, -1
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds %struct.B, ptr %0, i64 %24, i32 3
  %26 = load i32, ptr %25, align 4, !tbaa !16
  %27 = sub i32 %11, %26
  %28 = tail call range(i32 0, -2147483648) i32 @llvm.abs.i32(i32 %27, i1 true)
  %29 = icmp samesign ult i32 %28, %13
  %30 = select i1 %29, i32 %23, i32 %17
  %31 = icmp eq i32 %23, %4
  br i1 %31, label %32, label %15, !llvm.loop !19

32:                                               ; preds = %21
  %33 = lshr i16 %5, 9
  %34 = zext nneg i16 %33 to i64
  %35 = add i64 %7, %34
  %36 = trunc i64 %35 to i32
  %37 = sext i32 %30 to i64
  %38 = getelementptr inbounds %struct.B, ptr %0, i64 %37, i32 3
  store i32 %36, ptr %38, align 4, !tbaa !16
  ret i32 %30
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.abs.i32(i32, i1 immarg) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !10, i64 24}
!7 = !{!"C", !8, i64 0, !10, i64 20, !10, i64 24}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"int", !8, i64 0}
!11 = !{!12, !13, i64 0}
!12 = !{!"A", !13, i64 0, !14, i64 8}
!13 = !{!"short", !8, i64 0}
!14 = !{!"long", !8, i64 0}
!15 = !{!12, !14, i64 8}
!16 = !{!17, !10, i64 12}
!17 = !{!"B", !10, i64 0, !10, i64 4, !10, i64 8, !10, i64 12, !10, i64 16}
!18 = !{!7, !10, i64 20}
!19 = distinct !{!19, !20}
!20 = !{!"llvm.loop.mustprogress"}
