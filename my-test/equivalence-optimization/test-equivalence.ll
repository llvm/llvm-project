; ModuleID = 'test-equivalence.c'
source_filename = "test-equivalence.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f1(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = and i32 %1, %0
  %4 = icmp eq i32 %3, %1
  %5 = sub i32 %0, %1
  %6 = select i1 %4, i32 %5, i32 0
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f2(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = and i32 %1, %0
  %4 = icmp eq i32 %3, %1
  %5 = xor i32 %1, %0
  %6 = select i1 %4, i32 %5, i32 0
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f3(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = and i32 %1, %0
  %4 = icmp eq i32 %3, %1
  %5 = xor i32 %1, -1
  %6 = and i32 %0, %5
  %7 = select i1 %4, i32 %6, i32 0
  ret i32 %7
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 34109cd26ae1b317d91c061500d9828fe6ebab0b)"}
