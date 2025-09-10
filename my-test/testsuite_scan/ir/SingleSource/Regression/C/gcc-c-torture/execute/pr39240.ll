; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr39240.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr39240.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@l1 = dso_local global i64 4294967292, align 8
@l2 = dso_local global i64 65532, align 8
@l3 = dso_local global i64 252, align 8
@l4 = dso_local global i64 -4, align 8
@l5 = dso_local global i64 -4, align 8
@l6 = dso_local global i64 -4, align 8

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 -2147483642, -2147483648) i32 @bar1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 6
  ret i32 %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i16 @bar2(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 6
  %3 = tail call fastcc i16 @foo2(i32 noundef %2)
  ret i16 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define internal fastcc noundef i16 @foo2(i32 noundef range(i32 -2147483642, -2147483648) %0) unnamed_addr #0 {
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i8 @bar3(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 6
  %3 = tail call fastcc i8 @foo3(i32 noundef %2)
  ret i8 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define internal fastcc noundef i8 @foo3(i32 noundef range(i32 -2147483642, -2147483648) %0) unnamed_addr #0 {
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 -2147483642, -2147483648) i32 @bar4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 6
  ret i32 %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i16 @bar5(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 6
  %3 = tail call fastcc i16 @foo5(i32 noundef %2)
  ret i16 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define internal fastcc noundef i16 @foo5(i32 noundef range(i32 -2147483642, -2147483648) %0) unnamed_addr #0 {
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i8 @bar6(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 6
  %3 = tail call fastcc i8 @foo6(i32 noundef %2)
  ret i8 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define internal fastcc noundef i8 @foo6(i32 noundef range(i32 -2147483642, -2147483648) %0) unnamed_addr #0 {
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call i32 @bar1(i32 noundef -10)
  %2 = zext i32 %1 to i64
  %3 = load volatile i64, ptr @l1, align 8, !tbaa !6
  %4 = icmp eq i64 %3, %2
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

6:                                                ; preds = %0
  %7 = tail call i16 @bar2(i32 noundef -10)
  %8 = zext i16 %7 to i64
  %9 = load volatile i64, ptr @l2, align 8, !tbaa !6
  %10 = icmp eq i64 %9, %8
  br i1 %10, label %12, label %11

11:                                               ; preds = %6
  tail call void @abort() #3
  unreachable

12:                                               ; preds = %6
  %13 = tail call i8 @bar3(i32 noundef -10)
  %14 = zext i8 %13 to i64
  %15 = load volatile i64, ptr @l3, align 8, !tbaa !6
  %16 = icmp eq i64 %15, %14
  br i1 %16, label %18, label %17

17:                                               ; preds = %12
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %12
  %19 = tail call i32 @bar4(i32 noundef -10)
  %20 = sext i32 %19 to i64
  %21 = load volatile i64, ptr @l4, align 8, !tbaa !6
  %22 = icmp eq i64 %21, %20
  br i1 %22, label %24, label %23

23:                                               ; preds = %18
  tail call void @abort() #3
  unreachable

24:                                               ; preds = %18
  %25 = tail call i16 @bar5(i32 noundef -10)
  %26 = sext i16 %25 to i64
  %27 = load volatile i64, ptr @l5, align 8, !tbaa !6
  %28 = icmp eq i64 %27, %26
  br i1 %28, label %30, label %29

29:                                               ; preds = %24
  tail call void @abort() #3
  unreachable

30:                                               ; preds = %24
  %31 = tail call i8 @bar6(i32 noundef -10)
  %32 = sext i8 %31 to i64
  %33 = load volatile i64, ptr @l6, align 8, !tbaa !6
  %34 = icmp eq i64 %33, %32
  br i1 %34, label %36, label %35

35:                                               ; preds = %30
  tail call void @abort() #3
  unreachable

36:                                               ; preds = %30
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
