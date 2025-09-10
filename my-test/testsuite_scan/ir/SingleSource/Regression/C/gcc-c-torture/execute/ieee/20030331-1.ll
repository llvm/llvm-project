; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/20030331-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/20030331-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@x = dso_local local_unnamed_addr global float -1.500000e+00, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local float @rintf() local_unnamed_addr #0 {
  %1 = load float, ptr @x, align 4, !tbaa !6
  %2 = tail call float @llvm.fabs.f32(float %1)
  %3 = fcmp olt float %2, 0x4160000000000000
  br i1 %3, label %4, label %17

4:                                                ; preds = %0
  %5 = fcmp ogt float %1, 0.000000e+00
  br i1 %5, label %6, label %9

6:                                                ; preds = %4
  %7 = fadd float %1, 0x4160000000000000
  %8 = fadd float %7, 0xC160000000000000
  br label %15

9:                                                ; preds = %4
  %10 = fcmp olt float %1, 0.000000e+00
  br i1 %10, label %11, label %17

11:                                               ; preds = %9
  %12 = fsub float 0x4160000000000000, %1
  %13 = fadd float %12, 0xC160000000000000
  %14 = fneg float %13
  br label %15

15:                                               ; preds = %11, %6
  %16 = phi float [ %8, %6 ], [ %14, %11 ]
  store float %16, ptr @x, align 4, !tbaa !6
  br label %17

17:                                               ; preds = %15, %9, %0
  %18 = phi float [ %1, %9 ], [ %1, %0 ], [ %16, %15 ]
  ret float %18
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load float, ptr @x, align 4, !tbaa !6
  %2 = tail call float @llvm.fabs.f32(float %1)
  %3 = fcmp olt float %2, 0x4160000000000000
  br i1 %3, label %4, label %17

4:                                                ; preds = %0
  %5 = fcmp ogt float %1, 0.000000e+00
  br i1 %5, label %6, label %9

6:                                                ; preds = %4
  %7 = fadd float %1, 0x4160000000000000
  %8 = fadd float %7, 0xC160000000000000
  br label %15

9:                                                ; preds = %4
  %10 = fcmp olt float %1, 0.000000e+00
  br i1 %10, label %11, label %17

11:                                               ; preds = %9
  %12 = fsub float 0x4160000000000000, %1
  %13 = fadd float %12, 0xC160000000000000
  %14 = fneg float %13
  br label %15

15:                                               ; preds = %11, %6
  %16 = phi float [ %8, %6 ], [ %14, %11 ]
  store float %16, ptr @x, align 4, !tbaa !6
  br label %17

17:                                               ; preds = %0, %9, %15
  %18 = phi float [ %1, %9 ], [ %1, %0 ], [ %16, %15 ]
  %19 = fcmp une float %18, -2.000000e+00
  br i1 %19, label %20, label %21

20:                                               ; preds = %17
  tail call void @abort() #5
  unreachable

21:                                               ; preds = %17
  tail call void @exit(i32 noundef 0) #5
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
