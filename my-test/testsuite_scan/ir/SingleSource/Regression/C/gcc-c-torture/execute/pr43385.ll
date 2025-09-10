; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr43385.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr43385.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@e = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %9, label %4, !prof !6

4:                                                ; preds = %2
  %5 = icmp eq i32 %1, 0
  br i1 %5, label %9, label %6

6:                                                ; preds = %4
  %7 = load i32, ptr @e, align 4, !tbaa !7
  %8 = add nsw i32 %7, 1
  store i32 %8, ptr @e, align 4, !tbaa !7
  br label %9

9:                                                ; preds = %6, %4, %2
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @bar(i32 noundef %0, i32 noundef %1) local_unnamed_addr #1 {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %6, label %4, !prof !6

4:                                                ; preds = %2
  %5 = icmp eq i32 %1, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %4, %2
  br label %7

7:                                                ; preds = %4, %6
  %8 = phi i32 [ 0, %6 ], [ 1, %4 ]
  ret i32 %8
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = tail call i32 asm "", "=r,0"(i32 0) #4, !srcloc !11
  %2 = add nsw i32 %1, 2
  %3 = add nsw i32 %1, 1
  tail call void @foo(i32 noundef %2, i32 noundef %3)
  %4 = load i32, ptr @e, align 4, !tbaa !7
  %5 = icmp eq i32 %4, 1
  br i1 %5, label %7, label %6

6:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

7:                                                ; preds = %0
  tail call void @foo(i32 noundef %2, i32 noundef %1)
  %8 = load i32, ptr @e, align 4, !tbaa !7
  %9 = icmp eq i32 %8, 1
  br i1 %9, label %11, label %10

10:                                               ; preds = %7
  tail call void @abort() #5
  unreachable

11:                                               ; preds = %7
  tail call void @foo(i32 noundef %3, i32 noundef %3)
  %12 = load i32, ptr @e, align 4, !tbaa !7
  %13 = icmp eq i32 %12, 2
  br i1 %13, label %15, label %14

14:                                               ; preds = %11
  tail call void @abort() #5
  unreachable

15:                                               ; preds = %11
  tail call void @foo(i32 noundef %3, i32 noundef %1)
  %16 = load i32, ptr @e, align 4, !tbaa !7
  %17 = icmp eq i32 %16, 2
  br i1 %17, label %19, label %18

18:                                               ; preds = %15
  tail call void @abort() #5
  unreachable

19:                                               ; preds = %15
  tail call void @foo(i32 noundef %1, i32 noundef %3)
  %20 = load i32, ptr @e, align 4, !tbaa !7
  %21 = icmp eq i32 %20, 2
  br i1 %21, label %23, label %22

22:                                               ; preds = %19
  tail call void @abort() #5
  unreachable

23:                                               ; preds = %19
  tail call void @foo(i32 noundef %1, i32 noundef %1)
  %24 = load i32, ptr @e, align 4, !tbaa !7
  %25 = icmp eq i32 %24, 2
  br i1 %25, label %27, label %26

26:                                               ; preds = %23
  tail call void @abort() #5
  unreachable

27:                                               ; preds = %23
  %28 = tail call i32 @bar(i32 noundef %2, i32 noundef %3)
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %31

30:                                               ; preds = %27
  tail call void @abort() #5
  unreachable

31:                                               ; preds = %27
  %32 = tail call i32 @bar(i32 noundef %2, i32 noundef %1)
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %35, label %34

34:                                               ; preds = %31
  tail call void @abort() #5
  unreachable

35:                                               ; preds = %31
  %36 = tail call i32 @bar(i32 noundef %3, i32 noundef %3)
  %37 = icmp eq i32 %36, 0
  br i1 %37, label %38, label %39

38:                                               ; preds = %35
  tail call void @abort() #5
  unreachable

39:                                               ; preds = %35
  %40 = tail call i32 @bar(i32 noundef %3, i32 noundef %1)
  %41 = icmp eq i32 %40, 0
  br i1 %41, label %43, label %42

42:                                               ; preds = %39
  tail call void @abort() #5
  unreachable

43:                                               ; preds = %39
  %44 = tail call i32 @bar(i32 noundef %1, i32 noundef %3)
  %45 = icmp eq i32 %44, 0
  br i1 %45, label %47, label %46

46:                                               ; preds = %43
  tail call void @abort() #5
  unreachable

47:                                               ; preds = %43
  %48 = tail call i32 @bar(i32 noundef %1, i32 noundef %1)
  %49 = icmp eq i32 %48, 0
  br i1 %49, label %51, label %50

50:                                               ; preds = %47
  tail call void @abort() #5
  unreachable

51:                                               ; preds = %47
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind memory(none) }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i64 328}
