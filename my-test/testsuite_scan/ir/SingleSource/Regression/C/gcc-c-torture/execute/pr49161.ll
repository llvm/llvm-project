; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr49161.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr49161.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @bar(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @c, align 4, !tbaa !6
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr @c, align 4, !tbaa !6
  %4 = icmp eq i32 %0, %2
  br i1 %4, label %6, label %5

5:                                                ; preds = %1
  tail call void @abort() #3
  unreachable

6:                                                ; preds = %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  switch i32 %0, label %8 [
    i32 3, label %2
    i32 4, label %2
    i32 6, label %4
  ]

2:                                                ; preds = %1, %1
  tail call void @bar(i32 noundef 0)
  %3 = icmp eq i32 %0, 4
  br i1 %3, label %5, label %6

4:                                                ; preds = %1
  tail call void @bar(i32 noundef -1)
  tail call void @bar(i32 noundef 0)
  tail call void @bar(i32 noundef 1)
  br label %5

5:                                                ; preds = %4, %2
  br label %6

6:                                                ; preds = %2, %5
  %7 = phi i32 [ -1, %5 ], [ 1, %2 ]
  tail call void @bar(i32 noundef %7)
  tail call void @bar(i32 noundef 2)
  br label %8

8:                                                ; preds = %1, %6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  tail call void @foo(i32 noundef 3)
  %1 = load i32, ptr @c, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 3
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %0
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

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
