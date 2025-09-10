; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58984.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58984.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local global i32 0, align 4
@c = dso_local local_unnamed_addr global ptr @a, align 8
@n = dso_local local_unnamed_addr global i32 0, align 4
@m = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @e, align 4, !tbaa !6
  %2 = icmp slt i32 %1, 2
  br i1 %2, label %3, label %7

3:                                                ; preds = %0
  %4 = load ptr, ptr @c, align 8, !tbaa !10
  %5 = load i32, ptr %4, align 4, !tbaa !6
  %6 = xor i32 %5, 1
  store i32 %6, ptr %4, align 4, !tbaa !6
  br label %7

7:                                                ; preds = %0, %3
  store i32 1, ptr @m, align 4, !tbaa !6
  %8 = load i32, ptr @a, align 4, !tbaa !6
  %9 = icmp eq i32 %8, 1
  br i1 %9, label %11, label %10

10:                                               ; preds = %7
  tail call void @abort() #2
  unreachable

11:                                               ; preds = %7
  store i32 0, ptr @e, align 4, !tbaa !6
  %12 = load ptr, ptr @c, align 8, !tbaa !10
  %13 = load i32, ptr %12, align 4, !tbaa !6
  %14 = xor i32 %13, 1
  store i32 %14, ptr %12, align 4, !tbaa !6
  %15 = load i32, ptr @m, align 4, !tbaa !6
  %16 = or i32 %15, 1
  store i32 %16, ptr @m, align 4, !tbaa !6
  %17 = load i32, ptr @a, align 4, !tbaa !6
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %11
  tail call void @abort() #2
  unreachable

20:                                               ; preds = %11
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
!11 = !{!"p1 int", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
