; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr64756.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr64756.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@tmp = dso_local global ptr null, align 8
@c = dso_local local_unnamed_addr global ptr @tmp, align 8
@a = dso_local global i32 0, align 4
@d = dso_local global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @f, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %22

3:                                                ; preds = %0
  %4 = load ptr, ptr @c, align 8, !tbaa !10
  %5 = load volatile i32, ptr @d, align 4, !tbaa !6
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %9

7:                                                ; preds = %3
  %8 = load volatile i32, ptr @d, align 4, !tbaa !6
  br label %9

9:                                                ; preds = %7, %3
  store ptr @a, ptr %4, align 8, !tbaa !14
  %10 = load ptr, ptr @tmp, align 8, !tbaa !14
  %11 = icmp eq ptr %10, @a
  br i1 %11, label %13, label %12

12:                                               ; preds = %18, %9
  tail call void @abort() #2
  unreachable

13:                                               ; preds = %9
  store volatile ptr null, ptr @tmp, align 8, !tbaa !14
  %14 = load volatile i32, ptr @d, align 4, !tbaa !6
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %16, label %18

16:                                               ; preds = %13
  %17 = load volatile i32, ptr @d, align 4, !tbaa !6
  br label %18

18:                                               ; preds = %16, %13
  store ptr @a, ptr %4, align 8, !tbaa !14
  %19 = load ptr, ptr @tmp, align 8, !tbaa !14
  %20 = icmp eq ptr %19, @a
  br i1 %20, label %21, label %12

21:                                               ; preds = %18
  store volatile ptr null, ptr @tmp, align 8, !tbaa !14
  store i32 1, ptr @f, align 4, !tbaa !6
  br label %22

22:                                               ; preds = %0, %21
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
!10 = !{!11, !11, i64 0}
!11 = !{!"p2 int", !12, i64 0}
!12 = !{!"any p2 pointer", !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 int", !13, i64 0}
