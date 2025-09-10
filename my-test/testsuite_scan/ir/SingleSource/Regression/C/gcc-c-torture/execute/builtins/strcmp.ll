; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strcmp.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strcmp.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@x = dso_local local_unnamed_addr global i32 7, align 4
@.str = private unnamed_addr constant [9 x i8] c"hi world\00", align 1
@bar = dso_local local_unnamed_addr global ptr @.str, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = load ptr, ptr @bar, align 8, !tbaa !6
  %2 = load i8, ptr %1, align 1
  %3 = icmp eq i8 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  tail call void @abort() #2
  unreachable

5:                                                ; preds = %0
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load i8, ptr %6, align 1
  %8 = icmp eq i8 %7, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %5
  tail call void @abort() #2
  unreachable

10:                                               ; preds = %5
  %11 = load i32, ptr @x, align 4, !tbaa !11
  %12 = add nsw i32 %11, -1
  store i32 %12, ptr @x, align 4, !tbaa !11
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds i8, ptr %1, i64 %13
  %15 = load i8, ptr %14, align 1
  %16 = icmp eq i8 %15, 0
  %17 = icmp ne i32 %12, 6
  %18 = or i1 %17, %16
  br i1 %18, label %19, label %20

19:                                               ; preds = %10
  tail call void @abort() #2
  unreachable

20:                                               ; preds = %10
  store i32 %11, ptr @x, align 4, !tbaa !11
  %21 = zext nneg i32 %11 to i64
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 %21
  %23 = load i8, ptr %22, align 1
  %24 = icmp eq i8 %23, 0
  br i1 %24, label %25, label %26

25:                                               ; preds = %20
  tail call void @abort() #2
  unreachable

26:                                               ; preds = %20
  ret void
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
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
