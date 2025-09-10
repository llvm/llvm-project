; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vector/sumarray-dbl.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vector/sumarray-dbl.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%union.Array = type { [100 x <8 x double>] }

@TheArray = dso_local local_unnamed_addr global %union.Array zeroinitializer, align 16
@.str = private unnamed_addr constant [25 x i8] c"%g %g %g %g %g %g %g %g\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = phi <2 x i32> [ <i32 0, i32 1>, %0 ], [ %12, %1 ]
  %4 = add <2 x i32> %3, splat (i32 2)
  %5 = uitofp nneg <2 x i32> %3 to <2 x double>
  %6 = uitofp nneg <2 x i32> %4 to <2 x double>
  %7 = fmul <2 x double> %5, splat (double 1.234500e+01)
  %8 = fmul <2 x double> %6, splat (double 1.234500e+01)
  %9 = getelementptr inbounds nuw double, ptr @TheArray, i64 %2
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store <2 x double> %7, ptr %9, align 16, !tbaa !6
  store <2 x double> %8, ptr %10, align 16, !tbaa !6
  %11 = add nuw i64 %2, 4
  %12 = add <2 x i32> %3, splat (i32 4)
  %13 = icmp eq i64 %11, 800
  br i1 %13, label %14, label %1, !llvm.loop !9

14:                                               ; preds = %1, %14
  %15 = phi i64 [ %20, %14 ], [ 0, %1 ]
  %16 = phi <8 x double> [ %19, %14 ], [ zeroinitializer, %1 ]
  %17 = getelementptr inbounds nuw <8 x double>, ptr @TheArray, i64 %15
  %18 = load <8 x double>, ptr %17, align 16, !tbaa !6
  %19 = fadd <8 x double> %16, %18
  %20 = add nuw nsw i64 %15, 1
  %21 = icmp eq i64 %20, 100
  br i1 %21, label %22, label %14, !llvm.loop !13

22:                                               ; preds = %14
  %23 = extractelement <8 x double> %19, i64 0
  %24 = extractelement <8 x double> %19, i64 1
  %25 = extractelement <8 x double> %19, i64 2
  %26 = extractelement <8 x double> %19, i64 3
  %27 = extractelement <8 x double> %19, i64 4
  %28 = extractelement <8 x double> %19, i64 5
  %29 = extractelement <8 x double> %19, i64 6
  %30 = extractelement <8 x double> %19, i64 7
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %23, double noundef %24, double noundef %25, double noundef %26, double noundef %27, double noundef %28, double noundef %29, double noundef %30)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!13 = distinct !{!13, !10}
