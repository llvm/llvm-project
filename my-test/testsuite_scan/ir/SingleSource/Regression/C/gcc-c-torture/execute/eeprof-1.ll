; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/eeprof-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/eeprof-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@last_fn_entered = dso_local local_unnamed_addr global ptr null, align 8
@entry_calls = dso_local local_unnamed_addr global i32 0, align 4
@exit_calls = dso_local local_unnamed_addr global i32 0, align 4
@last_fn_exited = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo() #0 {
  %1 = load ptr, ptr @last_fn_entered, align 8, !tbaa !6
  %2 = icmp eq ptr %1, @foo
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %0
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @nfoo() local_unnamed_addr #0 {
  %1 = load i32, ptr @entry_calls, align 4, !tbaa !10
  %2 = icmp eq i32 %1, 2
  %3 = load i32, ptr @exit_calls, align 4
  %4 = icmp eq i32 %3, 2
  %5 = select i1 %2, i1 %4, i1 false
  br i1 %5, label %7, label %6

6:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

7:                                                ; preds = %0
  %8 = load ptr, ptr @last_fn_entered, align 8, !tbaa !6
  %9 = icmp eq ptr %8, @foo
  br i1 %9, label %11, label %10

10:                                               ; preds = %7
  tail call void @abort() #3
  unreachable

11:                                               ; preds = %7
  %12 = load ptr, ptr @last_fn_exited, align 8, !tbaa !6
  %13 = icmp eq ptr %12, @foo2
  br i1 %13, label %15, label %14

14:                                               ; preds = %11
  tail call void @abort() #3
  unreachable

15:                                               ; preds = %11
  tail call void @foo()
  %16 = load i32, ptr @entry_calls, align 4, !tbaa !10
  %17 = icmp eq i32 %16, 3
  %18 = load i32, ptr @exit_calls, align 4
  %19 = icmp eq i32 %18, 3
  %20 = select i1 %17, i1 %19, i1 false
  br i1 %20, label %22, label %21

21:                                               ; preds = %15
  tail call void @abort() #3
  unreachable

22:                                               ; preds = %15
  %23 = load ptr, ptr @last_fn_entered, align 8, !tbaa !6
  %24 = icmp eq ptr %23, @foo
  br i1 %24, label %26, label %25

25:                                               ; preds = %22
  tail call void @abort() #3
  unreachable

26:                                               ; preds = %22
  %27 = load ptr, ptr @last_fn_exited, align 8, !tbaa !6
  %28 = icmp eq ptr %27, @foo
  br i1 %28, label %30, label %29

29:                                               ; preds = %26
  tail call void @abort() #3
  unreachable

30:                                               ; preds = %26
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @entry_calls, align 4, !tbaa !10
  %2 = icmp eq i32 %1, 0
  %3 = load i32, ptr @exit_calls, align 4
  %4 = icmp eq i32 %3, 0
  %5 = select i1 %2, i1 %4, i1 false
  br i1 %5, label %7, label %6

6:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

7:                                                ; preds = %0
  tail call void @foo2()
  %8 = load i32, ptr @entry_calls, align 4, !tbaa !10
  %9 = icmp eq i32 %8, 2
  %10 = load i32, ptr @exit_calls, align 4
  %11 = icmp eq i32 %10, 2
  %12 = select i1 %9, i1 %11, i1 false
  br i1 %12, label %14, label %13

13:                                               ; preds = %7
  tail call void @abort() #3
  unreachable

14:                                               ; preds = %7
  %15 = load ptr, ptr @last_fn_entered, align 8, !tbaa !6
  %16 = icmp eq ptr %15, @foo
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %14
  %19 = load ptr, ptr @last_fn_exited, align 8, !tbaa !6
  %20 = icmp eq ptr %19, @foo2
  br i1 %20, label %22, label %21

21:                                               ; preds = %18
  tail call void @abort() #3
  unreachable

22:                                               ; preds = %18
  tail call void @nfoo()
  %23 = load i32, ptr @entry_calls, align 4, !tbaa !10
  %24 = icmp eq i32 %23, 3
  %25 = load i32, ptr @exit_calls, align 4
  %26 = icmp eq i32 %25, 3
  %27 = select i1 %24, i1 %26, i1 false
  br i1 %27, label %29, label %28

28:                                               ; preds = %22
  tail call void @abort() #3
  unreachable

29:                                               ; preds = %22
  %30 = load ptr, ptr @last_fn_entered, align 8, !tbaa !6
  %31 = icmp eq ptr %30, @foo
  br i1 %31, label %33, label %32

32:                                               ; preds = %29
  tail call void @abort() #3
  unreachable

33:                                               ; preds = %29
  ret i32 0
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @__cyg_profile_func_enter(ptr noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #2 {
  %3 = load i32, ptr @entry_calls, align 4, !tbaa !10
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @entry_calls, align 4, !tbaa !10
  store ptr %0, ptr @last_fn_entered, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @__cyg_profile_func_exit(ptr noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #2 {
  %3 = load i32, ptr @exit_calls, align 4, !tbaa !10
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @exit_calls, align 4, !tbaa !10
  store ptr %0, ptr @last_fn_exited, align 8, !tbaa !6
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define internal void @foo2() #0 {
  %1 = load i32, ptr @entry_calls, align 4, !tbaa !10
  %2 = icmp eq i32 %1, 1
  %3 = load i32, ptr @exit_calls, align 4
  %4 = icmp eq i32 %3, 0
  %5 = select i1 %2, i1 %4, i1 false
  br i1 %5, label %7, label %6

6:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

7:                                                ; preds = %0
  %8 = load ptr, ptr @last_fn_entered, align 8, !tbaa !6
  %9 = icmp eq ptr %8, @foo2
  br i1 %9, label %11, label %10

10:                                               ; preds = %7
  tail call void @abort() #3
  unreachable

11:                                               ; preds = %7
  tail call void @foo()
  %12 = load i32, ptr @entry_calls, align 4, !tbaa !10
  %13 = icmp eq i32 %12, 2
  %14 = load i32, ptr @exit_calls, align 4
  %15 = icmp eq i32 %14, 1
  %16 = select i1 %13, i1 %15, i1 false
  br i1 %16, label %18, label %17

17:                                               ; preds = %11
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %11
  %19 = load ptr, ptr @last_fn_entered, align 8, !tbaa !6
  %20 = icmp eq ptr %19, @foo
  br i1 %20, label %22, label %21

21:                                               ; preds = %18
  tail call void @abort() #3
  unreachable

22:                                               ; preds = %18
  %23 = load ptr, ptr @last_fn_exited, align 8, !tbaa !6
  %24 = icmp eq ptr %23, @foo
  br i1 %24, label %26, label %25

25:                                               ; preds = %22
  tail call void @abort() #3
  unreachable

26:                                               ; preds = %22
  ret void
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
