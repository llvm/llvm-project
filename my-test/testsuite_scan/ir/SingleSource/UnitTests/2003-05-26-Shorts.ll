; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-05-26-Shorts.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-05-26-Shorts.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [42 x i8] c"   ui = %u (0x%x)\09\09UL-ui = %lld (0x%llx)\0A\00", align 1
@.str.1 = private unnamed_addr constant [44 x i8] c"ui*ui = %u (0x%x)\09  UL/ui = %lld (0x%llx)\0A\0A\00", align 1
@.str.2 = private unnamed_addr constant [39 x i8] c"    i = %d (0x%x)\09L-i = %lld (0x%llx)\0A\00", align 1
@.str.3 = private unnamed_addr constant [41 x i8] c" i* i = %d (0x%x)\09L/ i = %lld (0x%llx)\0A\0A\00", align 1
@.str.4 = private unnamed_addr constant [42 x i8] c"us    = %u (0x%x)\09\09UL-us = %lld (0x%llx)\0A\00", align 1
@.str.5 = private unnamed_addr constant [44 x i8] c"us*us = %u (0x%x)\09  UL/us = %lld (0x%llx)\0A\0A\00", align 1
@.str.6 = private unnamed_addr constant [39 x i8] c" s    = %d (0x%x)\09L-s = %lld (0x%llx)\0A\00", align 1
@.str.7 = private unnamed_addr constant [41 x i8] c" s* s = %d (0x%x)\09L/ s = %lld (0x%llx)\0A\0A\00", align 1
@.str.8 = private unnamed_addr constant [42 x i8] c"ub    = %u (0x%x)\09\09UL-ub = %lld (0x%llx)\0A\00", align 1
@.str.9 = private unnamed_addr constant [43 x i8] c"ub*ub = %u (0x%x)\09\09UL/ub = %lld (0x%llx)\0A\0A\00", align 1
@.str.10 = private unnamed_addr constant [40 x i8] c" b    = %d (0x%x)\09\09L-b = %lld (0x%llx)\0A\00", align 1
@.str.11 = private unnamed_addr constant [42 x i8] c" b* b = %d (0x%x)\09\09\09L/b = %lld (0x%llx)\0A\0A\00", align 1
@.str.13 = private unnamed_addr constant [41 x i8] c"uiprod = %u (0x%x)\09  uidiv = %u (0x%x)\0A\0A\00", align 1
@.str.14 = private unnamed_addr constant [41 x i8] c"usprod = %u (0x%x)\09  usdiv = %u (0x%x)\0A\0A\00", align 1
@.str.15 = private unnamed_addr constant [41 x i8] c"ubprod = %u (0x%x)\09  ubdiv = %u (0x%x)\0A\0A\00", align 1
@.str.16 = private unnamed_addr constant [41 x i8] c" iprod = %u (0x%x)\09   idiv = %u (0x%x)\0A\0A\00", align 1
@.str.17 = private unnamed_addr constant [41 x i8] c" sprod = %u (0x%x)\09   sdiv = %u (0x%x)\0A\0A\00", align 1
@.str.18 = private unnamed_addr constant [41 x i8] c" bprod = %u (0x%x)\09   bdiv = %u (0x%x)\0A\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i64 @getL() local_unnamed_addr #0 {
  ret i64 -5787213826675591005
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef -976897885, i32 noundef -976897885, i64 noundef -5787213829993660416, i64 noundef -5787213829993660416)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef -1912031287, i32 noundef -1912031287, i64 noundef 3815330145, i64 noundef 3815330145)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef -976897885, i32 noundef -976897885, i64 noundef -5787213825698693120, i64 noundef -5787213825698693120)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef -1912031287, i32 noundef -1912031287, i64 noundef 5924072429, i64 noundef 5924072429)
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef 47267, i32 noundef 47267, i64 noundef -5787213826675638272, i64 noundef -5787213826675638272)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef -2060798007, i32 noundef -2060798007, i64 noundef 267830203885035, i64 noundef 267830203885035)
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef -18269, i32 noundef -18269, i64 noundef -5787213826675572736, i64 noundef -5787213826675572736)
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef 333756361, i32 noundef 333756361, i64 noundef 316777810864064, i64 noundef 316777810864064)
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef 163, i32 noundef 163, i64 noundef -5787213826675591168, i64 noundef -5787213826675591168)
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef 26569, i32 noundef 26569, i64 noundef 77665829736404666, i64 noundef 77665829736404666)
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef -93, i32 noundef -93, i64 noundef -5787213826675590912, i64 noundef -5787213826675590912)
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef 8649, i32 noundef 8649, i64 noundef 62228105663178397, i64 noundef 62228105663178397)
  %15 = tail call i32 @putchar(i32 10)
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef 41764483, i32 noundef 41764483, i32 noundef -512719167, i32 noundef -512719167)
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef 18051, i32 noundef 18051, i32 noundef 50617, i32 noundef 50617)
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i32 noundef 131, i32 noundef 131, i32 noundef 36, i32 noundef 36)
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16, i32 noundef 41764483, i32 noundef 41764483, i32 noundef -1918011031, i32 noundef -1918011031)
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.17, i32 noundef 18051, i32 noundef 18051, i32 noundef 4096, i32 noundef 4096)
  %21 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.18, i32 noundef -125, i32 noundef -125, i32 noundef 73, i32 noundef 73)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
