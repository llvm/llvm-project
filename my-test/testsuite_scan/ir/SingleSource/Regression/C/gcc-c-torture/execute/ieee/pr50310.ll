; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/pr50310.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/pr50310.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@s1 = dso_local local_unnamed_addr global [4 x double] zeroinitializer, align 16
@s2 = dso_local local_unnamed_addr global [4 x double] zeroinitializer, align 16
@s3 = dso_local local_unnamed_addr global [64 x double] zeroinitializer, align 16
@main.masks = internal unnamed_addr constant [8 x i32] [i32 2, i32 6, i32 1, i32 5, i32 3, i32 8, i32 2, i32 1], align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  %1 = load <2 x double>, ptr @s1, align 16, !tbaa !6
  %2 = load <2 x double>, ptr @s2, align 16, !tbaa !6
  %3 = fcmp ogt <2 x double> %1, %2
  %4 = select <2 x i1> %3, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %4, ptr @s3, align 16, !tbaa !6
  %5 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 16), align 16, !tbaa !6
  %6 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @s2, i64 16), align 16, !tbaa !6
  %7 = fcmp ogt <2 x double> %5, %6
  %8 = select <2 x i1> %7, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %8, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 16), align 16, !tbaa !6
  %9 = fcmp ule <2 x double> %1, %2
  %10 = select <2 x i1> %9, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %10, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 32), align 16, !tbaa !6
  %11 = fcmp ule <2 x double> %5, %6
  %12 = select <2 x i1> %11, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %12, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 48), align 16, !tbaa !6
  %13 = fcmp oge <2 x double> %1, %2
  %14 = select <2 x i1> %13, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %14, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 64), align 16, !tbaa !6
  %15 = fcmp oge <2 x double> %5, %6
  %16 = select <2 x i1> %15, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %16, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 80), align 16, !tbaa !6
  %17 = fcmp ult <2 x double> %1, %2
  %18 = select <2 x i1> %17, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %18, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 96), align 16, !tbaa !6
  %19 = fcmp ult <2 x double> %5, %6
  %20 = select <2 x i1> %19, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %20, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 112), align 16, !tbaa !6
  %21 = fcmp olt <2 x double> %1, %2
  %22 = select <2 x i1> %21, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %22, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 128), align 16, !tbaa !6
  %23 = fcmp olt <2 x double> %5, %6
  %24 = select <2 x i1> %23, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %24, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 144), align 16, !tbaa !6
  %25 = fcmp uge <2 x double> %1, %2
  %26 = select <2 x i1> %25, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %26, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 160), align 16, !tbaa !6
  %27 = fcmp uge <2 x double> %5, %6
  %28 = select <2 x i1> %27, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %28, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 176), align 16, !tbaa !6
  %29 = fcmp ole <2 x double> %1, %2
  %30 = select <2 x i1> %29, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %30, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 192), align 16, !tbaa !6
  %31 = fcmp ole <2 x double> %5, %6
  %32 = select <2 x i1> %31, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %32, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 208), align 16, !tbaa !6
  %33 = fcmp ugt <2 x double> %1, %2
  %34 = select <2 x i1> %33, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %34, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 224), align 16, !tbaa !6
  %35 = fcmp ugt <2 x double> %5, %6
  %36 = select <2 x i1> %35, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %36, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 240), align 16, !tbaa !6
  %37 = load <2 x double>, ptr @s1, align 16, !tbaa !6
  %38 = load <2 x double>, ptr @s2, align 16, !tbaa !6
  %39 = extractelement <2 x double> %37, i64 0
  %40 = extractelement <2 x double> %38, i64 0
  %41 = fcmp one double %39, %40
  %42 = select i1 %41, double -1.000000e+00, double 0.000000e+00
  store double %42, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 256), align 16, !tbaa !6
  %43 = extractelement <2 x double> %37, i64 1
  %44 = extractelement <2 x double> %38, i64 1
  %45 = fcmp one double %43, %44
  %46 = select i1 %45, double -1.000000e+00, double 0.000000e+00
  store double %46, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 264), align 8, !tbaa !6
  %47 = fcmp ueq double %39, %40
  %48 = select i1 %47, double -1.000000e+00, double 0.000000e+00
  store double %48, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 288), align 16, !tbaa !6
  %49 = fcmp ueq double %43, %44
  %50 = select i1 %49, double -1.000000e+00, double 0.000000e+00
  store double %50, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 296), align 8, !tbaa !6
  %51 = fcmp uno double %39, %40
  %52 = select i1 %51, double -1.000000e+00, double 0.000000e+00
  store double %52, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 320), align 16, !tbaa !6
  %53 = fcmp uno double %43, %44
  %54 = select i1 %53, double -1.000000e+00, double 0.000000e+00
  store double %54, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 328), align 8, !tbaa !6
  %55 = fcmp ord double %39, %40
  %56 = select i1 %55, double -1.000000e+00, double 0.000000e+00
  store double %56, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 352), align 16, !tbaa !6
  %57 = fcmp ord double %43, %44
  %58 = select i1 %57, double -1.000000e+00, double 0.000000e+00
  store double %58, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 360), align 8, !tbaa !6
  %59 = fcmp ogt <2 x double> %37, %38
  %60 = select <2 x i1> %59, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %60, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 384), align 16, !tbaa !6
  %61 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 16), align 16, !tbaa !6
  %62 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @s2, i64 16), align 16, !tbaa !6
  %63 = extractelement <2 x double> %61, i64 0
  %64 = extractelement <2 x double> %62, i64 0
  %65 = fcmp one double %63, %64
  %66 = select i1 %65, double -1.000000e+00, double 0.000000e+00
  store double %66, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 272), align 16, !tbaa !6
  %67 = extractelement <2 x double> %61, i64 1
  %68 = extractelement <2 x double> %62, i64 1
  %69 = fcmp one double %67, %68
  %70 = select i1 %69, double -1.000000e+00, double 0.000000e+00
  store double %70, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 280), align 8, !tbaa !6
  %71 = fcmp ueq double %63, %64
  %72 = select i1 %71, double -1.000000e+00, double 0.000000e+00
  store double %72, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 304), align 16, !tbaa !6
  %73 = fcmp ueq double %67, %68
  %74 = select i1 %73, double -1.000000e+00, double 0.000000e+00
  store double %74, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 312), align 8, !tbaa !6
  %75 = fcmp uno double %63, %64
  %76 = select i1 %75, double -1.000000e+00, double 0.000000e+00
  store double %76, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 336), align 16, !tbaa !6
  %77 = fcmp uno double %67, %68
  %78 = select i1 %77, double -1.000000e+00, double 0.000000e+00
  store double %78, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 344), align 8, !tbaa !6
  %79 = fcmp ord double %63, %64
  %80 = select i1 %79, double -1.000000e+00, double 0.000000e+00
  store double %80, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 368), align 16, !tbaa !6
  %81 = fcmp ord double %67, %68
  %82 = select i1 %81, double -1.000000e+00, double 0.000000e+00
  store double %82, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 376), align 8, !tbaa !6
  %83 = fcmp ogt <2 x double> %61, %62
  %84 = select <2 x i1> %83, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %84, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 400), align 16, !tbaa !6
  %85 = fcmp ole <2 x double> %37, %38
  %86 = select <2 x i1> %85, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %86, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 416), align 16, !tbaa !6
  %87 = fcmp ole <2 x double> %61, %62
  %88 = select <2 x i1> %87, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %88, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 432), align 16, !tbaa !6
  %89 = fcmp olt <2 x double> %37, %38
  %90 = select <2 x i1> %89, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %90, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 448), align 16, !tbaa !6
  %91 = fcmp olt <2 x double> %61, %62
  %92 = select <2 x i1> %91, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %92, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 464), align 16, !tbaa !6
  %93 = fcmp oge <2 x double> %37, %38
  %94 = select <2 x i1> %93, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %94, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 480), align 16, !tbaa !6
  %95 = fcmp oge <2 x double> %61, %62
  %96 = select <2 x i1> %95, <2 x double> splat (double -1.000000e+00), <2 x double> zeroinitializer
  store <2 x double> %96, ptr getelementptr inbounds nuw (i8, ptr @s3, i64 496), align 16, !tbaa !6
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store <2 x double> <double 5.000000e+00, double 6.000000e+00>, ptr @s1, align 16, !tbaa !6
  store <2 x double> <double 5.000000e+00, double 0x7FF8000000000000>, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 16), align 16, !tbaa !6
  store <2 x double> <double 6.000000e+00, double 5.000000e+00>, ptr @s2, align 16, !tbaa !6
  store <2 x double> splat (double 5.000000e+00), ptr getelementptr inbounds nuw (i8, ptr @s2, i64 16), align 16, !tbaa !6
  tail call void asm sideeffect "", "~{memory}"() #3, !srcloc !10
  tail call void @foo()
  tail call void asm sideeffect "", "~{memory}"() #3, !srcloc !11
  br label %1

1:                                                ; preds = %0, %27
  %2 = phi i64 [ 0, %0 ], [ %28, %27 ]
  %3 = icmp samesign ugt i64 %2, 47
  %4 = trunc nuw nsw i64 %2 to i32
  %5 = and i32 %4, 3
  %6 = icmp eq i32 %5, 3
  %7 = and i1 %3, %6
  %8 = getelementptr inbounds nuw double, ptr @s3, i64 %2
  %9 = load double, ptr %8, align 8, !tbaa !6
  br i1 %7, label %10, label %13

10:                                               ; preds = %1
  %11 = fcmp une double %9, 0.000000e+00
  br i1 %11, label %12, label %27

12:                                               ; preds = %10
  tail call void @abort() #4
  unreachable

13:                                               ; preds = %1
  %14 = shl nuw nsw i32 1, %5
  %15 = lshr i64 %2, 3
  %16 = and i64 %15, 536870911
  %17 = getelementptr inbounds nuw i32, ptr @main.masks, i64 %16
  %18 = load i32, ptr %17, align 4, !tbaa !12
  %19 = shl i32 %4, 29
  %20 = ashr i32 %19, 31
  %21 = xor i32 %18, %20
  %22 = and i32 %21, %14
  %23 = icmp eq i32 %22, 0
  %24 = select i1 %23, double 0.000000e+00, double -1.000000e+00
  %25 = fcmp une double %9, %24
  br i1 %25, label %26, label %27

26:                                               ; preds = %13
  tail call void @abort() #4
  unreachable

27:                                               ; preds = %10, %13
  %28 = add nuw nsw i64 %2, 1
  %29 = icmp eq i64 %28, 64
  br i1 %29, label %30, label %1, !llvm.loop !14

30:                                               ; preds = %27
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
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
!7 = !{!"double", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{i64 1767}
!11 = !{i64 1813}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !8, i64 0}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
