; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/pi.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/pi.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str.1 = private unnamed_addr constant [45 x i8] c" x = %9.6f    y = %12.2f  low = %8d j = %7d\0A\00", align 1
@.str.2 = private unnamed_addr constant [37 x i8] c"Pi = %9.6f ztot = %12.2f itot = %8d\0A\00", align 1
@str = private unnamed_addr constant [15 x i8] c"Starting PI...\00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @myadd(ptr noundef captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load float, ptr %0, align 4, !tbaa !6
  %4 = load float, ptr %1, align 4, !tbaa !6
  %5 = fadd float %3, %4
  store float %5, ptr %0, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %4

4:                                                ; preds = %2, %4
  %5 = phi float [ 5.813000e+03, %2 ], [ %18, %4 ]
  %6 = phi i64 [ 1, %2 ], [ %26, %4 ]
  %7 = phi i64 [ 1907, %2 ], [ %11, %4 ]
  %8 = phi i64 [ 1, %2 ], [ %25, %4 ]
  %9 = phi float [ 0.000000e+00, %2 ], [ %22, %4 ]
  %10 = mul nuw nsw i64 %7, 27611
  %11 = urem i64 %10, 74383
  %12 = uitofp nneg i64 %11 to float
  %13 = fdiv float %12, 7.438300e+04
  %14 = fmul float %5, 1.307000e+03
  %15 = fdiv float %14, 5.471000e+03
  %16 = fptosi float %15 to i64
  %17 = sitofp i64 %16 to float
  %18 = tail call float @llvm.fmuladd.f32(float %17, float -5.471000e+03, float %14)
  %19 = fdiv float %18, 5.471000e+03
  %20 = fmul float %19, %19
  %21 = tail call float @llvm.fmuladd.f32(float %13, float %13, float %20)
  %22 = fadd float %9, %21
  %23 = fcmp ole float %21, 1.000000e+00
  %24 = zext i1 %23 to i64
  %25 = add nuw nsw i64 %8, %24
  %26 = add nuw nsw i64 %6, 1
  %27 = icmp eq i64 %26, 40000001
  br i1 %27, label %28, label %4, !llvm.loop !10

28:                                               ; preds = %4
  %29 = fpext float %13 to double
  %30 = fpext float %19 to double
  %31 = trunc i64 %25 to i32
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %29, double noundef %30, i32 noundef %31, i32 noundef 40000001)
  %33 = uitofp nneg i64 %25 to float
  %34 = fpext float %33 to double
  %35 = fmul double %34, 4.000000e+00
  %36 = fdiv double %35, 4.000000e+07
  %37 = fptrunc double %36 to float
  %38 = fpext float %37 to double
  %39 = fpext float %22 to double
  %40 = fmul double %39, 0.000000e+00
  %41 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %38, double noundef %40, i32 noundef 40000000)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
