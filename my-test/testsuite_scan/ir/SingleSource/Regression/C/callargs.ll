; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/callargs.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/callargs.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str.1 = private unnamed_addr constant [29 x i8] c"\09Args 1-5  : %d %f %c %f %c\0A\00", align 1
@.str.2 = private unnamed_addr constant [29 x i8] c"\09Args 6-10 : %d %f %c %f %c\0A\00", align 1
@.str.3 = private unnamed_addr constant [29 x i8] c"\09Args 11-14: %d %f %c %f %c\0A\00", align 1
@str = private unnamed_addr constant [35 x i8] c"\0AprintArgsNoRet with 15 arguments:\00", align 4

; Function Attrs: nofree nounwind uwtable
define dso_local void @printArgsNoRet(i32 noundef %0, float noundef %1, i8 noundef %2, double noundef %3, ptr noundef readonly captures(none) %4, i32 noundef %5, float noundef %6, i8 noundef %7, double noundef %8, ptr noundef readonly captures(none) %9, i32 noundef %10, float noundef %11, i8 noundef %12, double noundef %13, ptr noundef readonly captures(none) %14) local_unnamed_addr #0 {
  %16 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %17 = fpext float %1 to double
  %18 = zext i8 %2 to i32
  %19 = load i8, ptr %4, align 1, !tbaa !6
  %20 = zext i8 %19 to i32
  %21 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %0, double noundef %17, i32 noundef %18, double noundef %3, i32 noundef %20)
  %22 = fpext float %6 to double
  %23 = zext i8 %7 to i32
  %24 = load i8, ptr %9, align 1, !tbaa !6
  %25 = zext i8 %24 to i32
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %5, double noundef %22, i32 noundef %23, double noundef %8, i32 noundef %25)
  %27 = fpext float %11 to double
  %28 = zext i8 %12 to i32
  %29 = load i8, ptr %14, align 1, !tbaa !6
  %30 = zext i8 %29 to i32
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %10, double noundef %27, i32 noundef %28, double noundef %13, i32 noundef %30)
  %32 = tail call i32 @putchar(i32 10)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 1, double noundef 0x4000CCCCC0000000, i32 noundef 99, double noundef 4.100000e+00, i32 noundef 101)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 6, double noundef 0x401C666660000000, i32 noundef 104, double noundef 9.100000e+00, i32 noundef 106)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef 11, double noundef 0x4028333340000000, i32 noundef 109, double noundef 1.410000e+01, i32 noundef 111)
  %7 = tail call i32 @putchar(i32 10)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind }

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
