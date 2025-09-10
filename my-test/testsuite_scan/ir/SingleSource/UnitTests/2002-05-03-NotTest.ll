; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2002-05-03-NotTest.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2002-05-03-NotTest.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [26 x i8] c"Bitwise Not: %d %d %d %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [32 x i8] c"Boolean Not: %d %d %d %d %d %d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @testBitWiseNot(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = xor i32 %0, -1
  %6 = xor i32 %1, -1
  %7 = xor i32 %2, -1
  %8 = xor i32 %3, -1
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @testBooleanNot(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %0, 0
  %6 = icmp sgt i32 %1, 0
  %7 = and i1 %5, %6
  %8 = xor i1 %7, true
  %9 = zext i1 %8 to i32
  %10 = icmp sgt i32 %2, 0
  %11 = and i1 %5, %10
  %12 = xor i1 %11, true
  %13 = zext i1 %12 to i32
  %14 = icmp sgt i32 %3, 0
  %15 = and i1 %5, %14
  %16 = xor i1 %15, true
  %17 = zext i1 %16 to i32
  %18 = and i1 %6, %10
  %19 = xor i1 %18, true
  %20 = zext i1 %19 to i32
  %21 = and i1 %6, %14
  %22 = xor i1 %21, true
  %23 = zext i1 %22 to i32
  %24 = and i1 %10, %14
  %25 = xor i1 %24, true
  %26 = zext i1 %25 to i32
  %27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %9, i32 noundef %13, i32 noundef %17, i32 noundef %20, i32 noundef %23, i32 noundef %26)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef -2, i32 noundef -3, i32 noundef 2, i32 noundef -6)
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 0, i32 noundef 1, i32 noundef 0, i32 noundef 1, i32 noundef 0, i32 noundef 1)
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
