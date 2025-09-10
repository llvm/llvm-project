; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr60960.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr60960.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -128) <4 x i8> @f1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = bitcast i32 %0 to <4 x i8>
  %3 = lshr <4 x i8> %2, splat (i8 1)
  ret <4 x i8> %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -128) <4 x i8> @f2(i32 noundef %0) local_unnamed_addr #0 {
  %2 = bitcast i32 %0 to <4 x i8>
  %3 = lshr <4 x i8> %2, splat (i8 1)
  ret <4 x i8> %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef <4 x i8> @f3(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = bitcast i32 %0 to <4 x i8>
  %4 = bitcast i32 %1 to <4 x i8>
  %5 = udiv <4 x i8> %3, %4
  ret <4 x i8> %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call <4 x i8> @f1(i32 noundef 84215045)
  %2 = bitcast <4 x i8> %1 to i32
  %3 = icmp eq i32 %2, 33686018
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %0
  %6 = tail call <4 x i8> @f2(i32 noundef 84215045)
  %7 = bitcast <4 x i8> %6 to i32
  %8 = icmp eq i32 %7, 33686018
  br i1 %8, label %10, label %9

9:                                                ; preds = %5
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %5
  %11 = tail call <4 x i8> @f3(i32 noundef 84215045, i32 noundef 33686018)
  %12 = bitcast <4 x i8> %11 to i32
  %13 = icmp eq i32 %12, 33686018
  br i1 %13, label %15, label %14

14:                                               ; preds = %10
  tail call void @abort() #3
  unreachable

15:                                               ; preds = %10
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
