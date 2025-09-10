; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68624.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68624.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@e = dso_local local_unnamed_addr global i32 1, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@j = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @h, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  %3 = load i32, ptr @c, align 4, !tbaa !6
  br i1 %2, label %4, label %17

4:                                                ; preds = %0
  %5 = icmp ne i32 %3, 0
  %6 = load i32, ptr @b, align 4
  %7 = icmp ne i32 %6, 0
  %8 = select i1 %5, i1 true, i1 %7
  %9 = zext i1 %8 to i32
  %10 = load i32, ptr @e, align 4, !tbaa !6
  %11 = srem i32 %9, %10
  store i32 %11, ptr @g, align 4, !tbaa !6
  %12 = icmp ne i32 %11, 0
  %13 = load i32, ptr @f, align 4
  %14 = icmp ne i32 %13, 0
  %15 = select i1 %12, i1 true, i1 %14
  %16 = select i1 %15, i1 %7, i1 false
  br i1 %16, label %17, label %19

17:                                               ; preds = %4, %0
  %18 = icmp eq i32 %3, -4
  br i1 %18, label %24, label %23

19:                                               ; preds = %4
  %20 = load i32, ptr @d, align 4, !tbaa !6
  store i32 %20, ptr @e, align 4, !tbaa !6
  %21 = icmp eq i32 %20, 0
  %22 = select i1 %21, i32 -4, i32 -5
  store i32 %22, ptr @c, align 4, !tbaa !6
  store i32 %22, ptr @j, align 4, !tbaa !6
  br i1 %21, label %24, label %23

23:                                               ; preds = %17, %19
  tail call void @abort() #2
  unreachable

24:                                               ; preds = %17, %19
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
