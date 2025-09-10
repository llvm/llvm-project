; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20000801-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20000801-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @foo(ptr noundef captures(address) %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 %3
  %5 = icmp eq i32 %1, 0
  br i1 %5, label %80, label %6

6:                                                ; preds = %2
  %7 = ptrtoint ptr %0 to i64
  %8 = add i64 %7, %3
  %9 = add i64 %7, 4
  %10 = tail call i64 @llvm.umax.i64(i64 %8, i64 %9)
  %11 = xor i64 %7, -1
  %12 = add i64 %10, %11
  %13 = lshr i64 %12, 2
  %14 = add nuw nsw i64 %13, 1
  %15 = icmp ult i64 %12, 28
  br i1 %15, label %72, label %16

16:                                               ; preds = %6
  %17 = icmp ult i64 %12, 124
  br i1 %17, label %51, label %18

18:                                               ; preds = %16
  %19 = and i64 %14, 9223372036854775776
  br label %20

20:                                               ; preds = %20, %18
  %21 = phi i64 [ 0, %18 ], [ %42, %20 ]
  %22 = shl i64 %21, 2
  %23 = getelementptr i8, ptr %0, i64 %22
  %24 = getelementptr i8, ptr %0, i64 %22
  %25 = getelementptr i8, ptr %24, i64 64
  %26 = load <64 x i8>, ptr %23, align 1, !tbaa !6
  %27 = shufflevector <64 x i8> %26, <64 x i8> poison, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28, i32 32, i32 36, i32 40, i32 44, i32 48, i32 52, i32 56, i32 60>
  %28 = shufflevector <64 x i8> %26, <64 x i8> poison, <16 x i32> <i32 1, i32 5, i32 9, i32 13, i32 17, i32 21, i32 25, i32 29, i32 33, i32 37, i32 41, i32 45, i32 49, i32 53, i32 57, i32 61>
  %29 = shufflevector <64 x i8> %26, <64 x i8> poison, <16 x i32> <i32 2, i32 6, i32 10, i32 14, i32 18, i32 22, i32 26, i32 30, i32 34, i32 38, i32 42, i32 46, i32 50, i32 54, i32 58, i32 62>
  %30 = shufflevector <64 x i8> %26, <64 x i8> poison, <16 x i32> <i32 3, i32 7, i32 11, i32 15, i32 19, i32 23, i32 27, i32 31, i32 35, i32 39, i32 43, i32 47, i32 51, i32 55, i32 59, i32 63>
  %31 = load <64 x i8>, ptr %25, align 1, !tbaa !6
  %32 = shufflevector <64 x i8> %31, <64 x i8> poison, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28, i32 32, i32 36, i32 40, i32 44, i32 48, i32 52, i32 56, i32 60>
  %33 = shufflevector <64 x i8> %31, <64 x i8> poison, <16 x i32> <i32 1, i32 5, i32 9, i32 13, i32 17, i32 21, i32 25, i32 29, i32 33, i32 37, i32 41, i32 45, i32 49, i32 53, i32 57, i32 61>
  %34 = shufflevector <64 x i8> %31, <64 x i8> poison, <16 x i32> <i32 2, i32 6, i32 10, i32 14, i32 18, i32 22, i32 26, i32 30, i32 34, i32 38, i32 42, i32 46, i32 50, i32 54, i32 58, i32 62>
  %35 = shufflevector <64 x i8> %31, <64 x i8> poison, <16 x i32> <i32 3, i32 7, i32 11, i32 15, i32 19, i32 23, i32 27, i32 31, i32 35, i32 39, i32 43, i32 47, i32 51, i32 55, i32 59, i32 63>
  %36 = shufflevector <16 x i8> %30, <16 x i8> %29, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %37 = shufflevector <16 x i8> %28, <16 x i8> %27, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %38 = shufflevector <32 x i8> %36, <32 x i8> %37, <64 x i32> <i32 0, i32 16, i32 32, i32 48, i32 1, i32 17, i32 33, i32 49, i32 2, i32 18, i32 34, i32 50, i32 3, i32 19, i32 35, i32 51, i32 4, i32 20, i32 36, i32 52, i32 5, i32 21, i32 37, i32 53, i32 6, i32 22, i32 38, i32 54, i32 7, i32 23, i32 39, i32 55, i32 8, i32 24, i32 40, i32 56, i32 9, i32 25, i32 41, i32 57, i32 10, i32 26, i32 42, i32 58, i32 11, i32 27, i32 43, i32 59, i32 12, i32 28, i32 44, i32 60, i32 13, i32 29, i32 45, i32 61, i32 14, i32 30, i32 46, i32 62, i32 15, i32 31, i32 47, i32 63>
  store <64 x i8> %38, ptr %23, align 1, !tbaa !6
  %39 = shufflevector <16 x i8> %35, <16 x i8> %34, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %40 = shufflevector <16 x i8> %33, <16 x i8> %32, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %41 = shufflevector <32 x i8> %39, <32 x i8> %40, <64 x i32> <i32 0, i32 16, i32 32, i32 48, i32 1, i32 17, i32 33, i32 49, i32 2, i32 18, i32 34, i32 50, i32 3, i32 19, i32 35, i32 51, i32 4, i32 20, i32 36, i32 52, i32 5, i32 21, i32 37, i32 53, i32 6, i32 22, i32 38, i32 54, i32 7, i32 23, i32 39, i32 55, i32 8, i32 24, i32 40, i32 56, i32 9, i32 25, i32 41, i32 57, i32 10, i32 26, i32 42, i32 58, i32 11, i32 27, i32 43, i32 59, i32 12, i32 28, i32 44, i32 60, i32 13, i32 29, i32 45, i32 61, i32 14, i32 30, i32 46, i32 62, i32 15, i32 31, i32 47, i32 63>
  store <64 x i8> %41, ptr %25, align 1, !tbaa !6
  %42 = add nuw i64 %21, 32
  %43 = icmp eq i64 %42, %19
  br i1 %43, label %44, label %20, !llvm.loop !9

44:                                               ; preds = %20
  %45 = icmp eq i64 %14, %19
  br i1 %45, label %80, label %46

46:                                               ; preds = %44
  %47 = shl i64 %19, 2
  %48 = getelementptr i8, ptr %0, i64 %47
  %49 = and i64 %14, 24
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %72, label %51

51:                                               ; preds = %46, %16
  %52 = phi i64 [ %19, %46 ], [ 0, %16 ]
  %53 = and i64 %14, 9223372036854775800
  %54 = shl i64 %53, 2
  %55 = getelementptr i8, ptr %0, i64 %54
  br label %56

56:                                               ; preds = %56, %51
  %57 = phi i64 [ %52, %51 ], [ %68, %56 ]
  %58 = shl i64 %57, 2
  %59 = getelementptr i8, ptr %0, i64 %58
  %60 = load <32 x i8>, ptr %59, align 1, !tbaa !6
  %61 = shufflevector <32 x i8> %60, <32 x i8> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %62 = shufflevector <32 x i8> %60, <32 x i8> poison, <8 x i32> <i32 1, i32 5, i32 9, i32 13, i32 17, i32 21, i32 25, i32 29>
  %63 = shufflevector <32 x i8> %60, <32 x i8> poison, <8 x i32> <i32 2, i32 6, i32 10, i32 14, i32 18, i32 22, i32 26, i32 30>
  %64 = shufflevector <32 x i8> %60, <32 x i8> poison, <8 x i32> <i32 3, i32 7, i32 11, i32 15, i32 19, i32 23, i32 27, i32 31>
  %65 = shufflevector <8 x i8> %64, <8 x i8> %63, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %66 = shufflevector <8 x i8> %62, <8 x i8> %61, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %67 = shufflevector <16 x i8> %65, <16 x i8> %66, <32 x i32> <i32 0, i32 8, i32 16, i32 24, i32 1, i32 9, i32 17, i32 25, i32 2, i32 10, i32 18, i32 26, i32 3, i32 11, i32 19, i32 27, i32 4, i32 12, i32 20, i32 28, i32 5, i32 13, i32 21, i32 29, i32 6, i32 14, i32 22, i32 30, i32 7, i32 15, i32 23, i32 31>
  store <32 x i8> %67, ptr %59, align 1, !tbaa !6
  %68 = add nuw i64 %57, 8
  %69 = icmp eq i64 %68, %53
  br i1 %69, label %70, label %56, !llvm.loop !13

70:                                               ; preds = %56
  %71 = icmp eq i64 %14, %53
  br i1 %71, label %80, label %72

72:                                               ; preds = %46, %70, %6
  %73 = phi ptr [ %0, %6 ], [ %48, %46 ], [ %55, %70 ]
  br label %74

74:                                               ; preds = %72, %74
  %75 = phi ptr [ %78, %74 ], [ %73, %72 ]
  %76 = load <4 x i8>, ptr %75, align 1, !tbaa !6
  %77 = shufflevector <4 x i8> %76, <4 x i8> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x i8> %77, ptr %75, align 1, !tbaa !6
  %78 = getelementptr inbounds nuw i8, ptr %75, i64 4
  %79 = icmp ult ptr %78, %4
  br i1 %79, label %74, label %80, !llvm.loop !14

80:                                               ; preds = %74, %44, %70, %2
  ret void
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void @exit(i32 noundef 0) #4
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #3

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { noreturn nounwind }

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
!13 = distinct !{!13, !10, !11, !12}
!14 = distinct !{!14, !10, !12, !11}
