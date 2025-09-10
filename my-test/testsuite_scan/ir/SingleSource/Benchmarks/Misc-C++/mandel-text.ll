; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/mandel-text.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/mandel-text.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@stdout = external local_unnamed_addr global ptr, align 8

; Function Attrs: mustprogress nofree norecurse nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %2

1:                                                ; preds = %10
  ret i32 0

2:                                                ; preds = %0, %10
  %3 = phi i32 [ 0, %0 ], [ %13, %10 ]
  %4 = uitofp nneg i32 %3 to double
  %5 = tail call double @llvm.fmuladd.f64(double %4, double 5.000000e-02, double -1.000000e+00)
  br label %6

6:                                                ; preds = %2, %43
  %7 = phi i32 [ 0, %2 ], [ %47, %43 ]
  %8 = uitofp nneg i32 %7 to double
  %9 = tail call double @llvm.fmuladd.f64(double %8, double 5.000000e-02, double -2.300000e+00)
  br label %18

10:                                               ; preds = %43
  %11 = load ptr, ptr @stdout, align 8, !tbaa !6
  %12 = tail call i32 @putc(i32 noundef 10, ptr noundef %11)
  %13 = add nuw nsw i32 %3, 1
  %14 = icmp eq i32 %13, 40
  br i1 %14, label %1, label %2, !llvm.loop !11

15:                                               ; preds = %35
  %16 = icmp samesign ult i32 %23, 100000
  %17 = select i1 %16, i32 88, i32 46
  br label %43

18:                                               ; preds = %38, %6
  %19 = phi i32 [ 0, %6 ], [ %39, %38 ]
  br label %20

20:                                               ; preds = %18, %28
  %21 = phi double [ %5, %18 ], [ %30, %28 ]
  %22 = phi double [ %9, %18 ], [ %32, %28 ]
  %23 = phi i32 [ 0, %18 ], [ %33, %28 ]
  %24 = fmul double %22, %22
  %25 = fmul double %21, %21
  %26 = fadd double %24, %25
  %27 = fcmp ule double %26, 4.000000e+00
  br i1 %27, label %28, label %35

28:                                               ; preds = %20
  %29 = fmul double %22, 2.000000e+00
  %30 = tail call double @llvm.fmuladd.f64(double %29, double %21, double %5)
  %31 = fsub double %24, %25
  %32 = fadd double %9, %31
  %33 = add nuw nsw i32 %23, 1
  %34 = icmp eq i32 %33, 255
  br i1 %34, label %40, label %20, !llvm.loop !13

35:                                               ; preds = %20
  %36 = add nuw nsw i32 %19, 1
  %37 = icmp eq i32 %36, 2000
  br i1 %37, label %15, label %38

38:                                               ; preds = %35, %40
  %39 = phi i32 [ %36, %35 ], [ %41, %40 ]
  br label %18, !llvm.loop !14

40:                                               ; preds = %28
  %41 = add nuw nsw i32 %19, 1
  %42 = icmp eq i32 %41, 2000
  br i1 %42, label %43, label %38

43:                                               ; preds = %40, %15
  %44 = phi i32 [ %17, %15 ], [ 88, %40 ]
  %45 = load ptr, ptr @stdout, align 8, !tbaa !6
  %46 = tail call i32 @putc(i32 noundef %44, ptr noundef %45)
  %47 = add nuw nsw i32 %7, 1
  %48 = icmp eq i32 %47, 78
  br i1 %48, label %10, label %6, !llvm.loop !15
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nofree nounwind
declare noundef i32 @putc(i32 noundef, ptr noundef captures(none)) local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS8_IO_FILE", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
!14 = distinct !{!14, !12}
!15 = distinct !{!15, !12}
