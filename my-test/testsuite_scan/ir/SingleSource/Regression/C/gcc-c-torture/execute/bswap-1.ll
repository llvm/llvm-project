; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/bswap-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/bswap-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i64 @g(i64 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i64 @llvm.bswap.i64(i64 %0)
  ret i64 %2
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.bswap.i64(i64) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i64 @f(i64 noundef %0) local_unnamed_addr #2 {
  %2 = tail call i64 @llvm.bswap.i64(i64 %0)
  ret i64 %2
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = tail call i64 @g(i64 noundef 18)
  %2 = icmp eq i64 %1, 1297036692682702848
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

4:                                                ; preds = %0
  %5 = tail call i64 @g(i64 noundef 4660)
  %6 = icmp eq i64 %5, 3752061439553044480
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %4
  %9 = tail call i64 @g(i64 noundef 1193046)
  %10 = icmp eq i64 %9, 6211609577260056576
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #5
  unreachable

12:                                               ; preds = %8
  %13 = tail call i64 @g(i64 noundef 305419896)
  %14 = icmp eq i64 %13, 8671175384462524416
  br i1 %14, label %16, label %15

15:                                               ; preds = %12
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %12
  %17 = tail call i64 @g(i64 noundef 78187493520)
  %18 = icmp eq i64 %17, -8036578753402372096
  br i1 %18, label %20, label %19

19:                                               ; preds = %16
  tail call void @abort() #5
  unreachable

20:                                               ; preds = %16
  %21 = tail call i64 @g(i64 noundef 20015998341138)
  %22 = icmp eq i64 %21, 1337701400965152768
  br i1 %22, label %24, label %23

23:                                               ; preds = %20
  tail call void @abort() #5
  unreachable

24:                                               ; preds = %20
  %25 = tail call i64 @g(i64 noundef 5124095575331380)
  %26 = icmp eq i64 %25, 3752220286069772800
  br i1 %26, label %28, label %27

27:                                               ; preds = %24
  tail call void @abort() #5
  unreachable

28:                                               ; preds = %24
  %29 = tail call i64 @g(i64 noundef 1311768467284833366)
  %30 = icmp eq i64 %29, 6211610197754262546
  br i1 %30, label %32, label %31

31:                                               ; preds = %28
  tail call void @abort() #5
  unreachable

32:                                               ; preds = %28
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
