; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2005-07-17-INT-To-FP.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2005-07-17-INT-To-FP.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@tests = dso_local local_unnamed_addr global [11 x i32] [i32 -2147483648, i32 -123456792, i32 -10, i32 -2, i32 -1, i32 0, i32 1, i32 2, i32 10, i32 123456792, i32 2147483520], align 4
@.str = private unnamed_addr constant [19 x i8] c"%d %f, %f, %f, %f\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %21, %1 ]
  %3 = trunc nuw nsw i64 %2 to i32
  %4 = trunc i64 %2 to i8
  %5 = shl nuw i8 %4, 2
  %6 = sitofp i8 %5 to double
  %7 = trunc i64 %2 to i16
  %8 = shl nuw i16 %7, 10
  %9 = sitofp i16 %8 to double
  %10 = trunc i64 %2 to i32
  %11 = shl i32 %10, 26
  %12 = sitofp i32 %11 to double
  %13 = shl nuw i64 %2, 58
  %14 = sitofp i64 %13 to double
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %3, double noundef %6, double noundef %9, double noundef %12, double noundef %14)
  %16 = uitofp i8 %5 to double
  %17 = uitofp i16 %8 to double
  %18 = uitofp i32 %11 to double
  %19 = uitofp i64 %13 to double
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %3, double noundef %16, double noundef %17, double noundef %18, double noundef %19)
  %21 = add nuw nsw i64 %2, 1
  %22 = icmp eq i64 %21, 64
  br i1 %22, label %23, label %1, !llvm.loop !6

23:                                               ; preds = %1
  %24 = load i32, ptr @tests, align 4, !tbaa !8
  %25 = uitofp i32 %24 to double
  %26 = sitofp i32 %24 to double
  %27 = sitofp i32 %24 to float
  %28 = fpext float %27 to double
  %29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0, double noundef %25, double noundef %26, double noundef %28, double noundef %28)
  %30 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 4), align 4, !tbaa !8
  %31 = uitofp i32 %30 to double
  %32 = sitofp i32 %30 to double
  %33 = sitofp i32 %30 to float
  %34 = fpext float %33 to double
  %35 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1, double noundef %31, double noundef %32, double noundef %34, double noundef %34)
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 8), align 4, !tbaa !8
  %37 = uitofp i32 %36 to double
  %38 = sitofp i32 %36 to double
  %39 = sitofp i32 %36 to float
  %40 = fpext float %39 to double
  %41 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 2, double noundef %37, double noundef %38, double noundef %40, double noundef %40)
  %42 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 12), align 4, !tbaa !8
  %43 = uitofp i32 %42 to double
  %44 = sitofp i32 %42 to double
  %45 = sitofp i32 %42 to float
  %46 = fpext float %45 to double
  %47 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 3, double noundef %43, double noundef %44, double noundef %46, double noundef %46)
  %48 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 16), align 4, !tbaa !8
  %49 = uitofp i32 %48 to double
  %50 = sitofp i32 %48 to double
  %51 = sitofp i32 %48 to float
  %52 = fpext float %51 to double
  %53 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 4, double noundef %49, double noundef %50, double noundef %52, double noundef %52)
  %54 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 20), align 4, !tbaa !8
  %55 = uitofp i32 %54 to double
  %56 = sitofp i32 %54 to double
  %57 = sitofp i32 %54 to float
  %58 = fpext float %57 to double
  %59 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 5, double noundef %55, double noundef %56, double noundef %58, double noundef %58)
  %60 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 24), align 4, !tbaa !8
  %61 = uitofp i32 %60 to double
  %62 = sitofp i32 %60 to double
  %63 = sitofp i32 %60 to float
  %64 = fpext float %63 to double
  %65 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 6, double noundef %61, double noundef %62, double noundef %64, double noundef %64)
  %66 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 28), align 4, !tbaa !8
  %67 = uitofp i32 %66 to double
  %68 = sitofp i32 %66 to double
  %69 = sitofp i32 %66 to float
  %70 = fpext float %69 to double
  %71 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 7, double noundef %67, double noundef %68, double noundef %70, double noundef %70)
  %72 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 32), align 4, !tbaa !8
  %73 = uitofp i32 %72 to double
  %74 = sitofp i32 %72 to double
  %75 = sitofp i32 %72 to float
  %76 = fpext float %75 to double
  %77 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 8, double noundef %73, double noundef %74, double noundef %76, double noundef %76)
  %78 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 36), align 4, !tbaa !8
  %79 = uitofp i32 %78 to double
  %80 = sitofp i32 %78 to double
  %81 = sitofp i32 %78 to float
  %82 = fpext float %81 to double
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 9, double noundef %79, double noundef %80, double noundef %82, double noundef %82)
  %84 = load i32, ptr getelementptr inbounds nuw (i8, ptr @tests, i64 40), align 4, !tbaa !8
  %85 = uitofp i32 %84 to double
  %86 = sitofp i32 %84 to double
  %87 = sitofp i32 %84 to float
  %88 = fpext float %87 to double
  %89 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 10, double noundef %85, double noundef %86, double noundef %88, double noundef %88)
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
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
