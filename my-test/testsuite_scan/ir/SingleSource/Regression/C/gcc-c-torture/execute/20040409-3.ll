; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20040409-3.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20040409-3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @test1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = xor i32 %0, 2147483647
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @test1u(i32 noundef %0) local_unnamed_addr #0 {
  %2 = xor i32 %0, 2147483647
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @test2u(i32 noundef %0) local_unnamed_addr #0 {
  %2 = xor i32 %0, 2147483647
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @test3u(i32 noundef %0) local_unnamed_addr #0 {
  %2 = xor i32 %0, 2147483647
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @test4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = xor i32 %0, 2147483647
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @test4u(i32 noundef %0) local_unnamed_addr #0 {
  %2 = xor i32 %0, 2147483647
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @test5u(i32 noundef %0) local_unnamed_addr #0 {
  %2 = xor i32 %0, 2147483647
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @test6u(i32 noundef %0) local_unnamed_addr #0 {
  %2 = xor i32 %0, 2147483647
  ret i32 %2
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test(i32 noundef %0, i32 noundef %1) local_unnamed_addr #1 {
  %3 = xor i32 %1, %0
  %4 = icmp eq i32 %3, 2147483647
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  tail call void @abort() #3
  unreachable

6:                                                ; preds = %2
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree nounwind uwtable
define dso_local void @testu(i32 noundef %0, i32 noundef %1) local_unnamed_addr #1 {
  %3 = xor i32 %1, %0
  %4 = icmp eq i32 %3, 2147483647
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  tail call void @abort() #3
  unreachable

6:                                                ; preds = %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
