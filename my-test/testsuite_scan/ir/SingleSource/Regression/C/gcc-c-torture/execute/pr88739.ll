; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr88739.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr88739.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v = dso_local global { { i32, i32, i32, i8, i8, i8, i8 } } { { i32, i32, i32, i8, i8, i8, i8 } { i32 0, i32 0, i32 0, i8 -119, i8 64, i8 0, i8 0 } }, align 4
@bar.i = internal unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @bar.i, align 4, !tbaa !6
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr @bar.i, align 4, !tbaa !6
  switch i32 %2, label %28 [
    i32 0, label %4
    i32 1, label %9
    i32 2, label %14
    i32 3, label %18
    i32 4, label %23
  ]

4:                                                ; preds = %1
  %5 = load volatile i32, ptr getelementptr inbounds nuw (i8, ptr @v, i64 12), align 4
  %6 = and i32 %5, 1073741823
  %7 = icmp eq i32 %0, %6
  br i1 %7, label %29, label %8

8:                                                ; preds = %4
  tail call void @abort() #2
  unreachable

9:                                                ; preds = %1
  %10 = load volatile i32, ptr getelementptr inbounds nuw (i8, ptr @v, i64 12), align 4
  %11 = lshr i32 %10, 30
  %12 = icmp eq i32 %0, %11
  br i1 %12, label %29, label %13

13:                                               ; preds = %9
  tail call void @abort() #2
  unreachable

14:                                               ; preds = %1
  %15 = load volatile i32, ptr getelementptr inbounds nuw (i8, ptr @v, i64 12), align 4, !tbaa !10
  %16 = icmp eq i32 %0, %15
  br i1 %16, label %29, label %17

17:                                               ; preds = %14
  tail call void @abort() #2
  unreachable

18:                                               ; preds = %1
  %19 = load volatile i16, ptr getelementptr inbounds nuw (i8, ptr @v, i64 12), align 4, !tbaa !10
  %20 = zext i16 %19 to i32
  %21 = icmp eq i32 %0, %20
  br i1 %21, label %29, label %22

22:                                               ; preds = %18
  tail call void @abort() #2
  unreachable

23:                                               ; preds = %1
  %24 = load volatile i16, ptr getelementptr inbounds nuw (i8, ptr @v, i64 14), align 2, !tbaa !10
  %25 = zext i16 %24 to i32
  %26 = icmp eq i32 %0, %25
  br i1 %26, label %29, label %27

27:                                               ; preds = %23
  tail call void @abort() #2
  unreachable

28:                                               ; preds = %1
  tail call void @abort() #2
  unreachable

29:                                               ; preds = %23, %18, %14, %9, %4
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 2
  tail call void @bar(i32 noundef %2)
  tail call void @bar(i32 noundef 0)
  tail call void @bar(i32 noundef %2)
  %3 = and i32 %2, 65535
  tail call void @bar(i32 noundef %3)
  %4 = lshr i32 %0, 18
  tail call void @bar(i32 noundef %4)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void @bar(i32 noundef 16521)
  tail call void @bar(i32 noundef 0)
  tail call void @bar(i32 noundef 16521)
  tail call void @bar(i32 noundef 16521)
  tail call void @bar(i32 noundef 0)
  ret i32 0
}

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
!10 = !{!8, !8, i64 0}
