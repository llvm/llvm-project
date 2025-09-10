; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/div.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/div.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [22 x i8] c"uDivTest1(42,3) = %u\0A\00", align 1
@.str.1 = private unnamed_addr constant [22 x i8] c"uDivTest2(42,3) = %u\0A\00", align 1
@.str.2 = private unnamed_addr constant [22 x i8] c"uDivTest3(42,3) = %u\0A\00", align 1
@.str.3 = private unnamed_addr constant [22 x i8] c"uDivTest4(42,3) = %u\0A\00", align 1
@.str.4 = private unnamed_addr constant [22 x i8] c"uDivTest5(42,3) = %u\0A\00", align 1
@.str.5 = private unnamed_addr constant [22 x i8] c"uDivTest6(42,3) = %u\0A\00", align 1
@.str.6 = private unnamed_addr constant [22 x i8] c"uDivTest7(42,3) = %u\0A\00", align 1
@.str.7 = private unnamed_addr constant [22 x i8] c"uDivTest8(42,3) = %u\0A\00", align 1
@.str.8 = private unnamed_addr constant [22 x i8] c"uDivTest9(42,3) = %u\0A\00", align 1
@.str.9 = private unnamed_addr constant [23 x i8] c"uDivTest10(42,3) = %u\0A\00", align 1
@.str.10 = private unnamed_addr constant [23 x i8] c"uDivTest11(42,3) = %u\0A\00", align 1
@.str.11 = private unnamed_addr constant [23 x i8] c"uDivTest12(42,3) = %u\0A\00", align 1
@.str.12 = private unnamed_addr constant [23 x i8] c"uDivTest13(42,3) = %u\0A\00", align 1
@.str.13 = private unnamed_addr constant [23 x i8] c"uDivTest14(42,3) = %u\0A\00", align 1
@.str.14 = private unnamed_addr constant [23 x i8] c"uDivTest15(42,3) = %u\0A\00", align 1
@.str.15 = private unnamed_addr constant [23 x i8] c"uDivTest16(42,3) = %u\0A\00", align 1
@.str.16 = private unnamed_addr constant [23 x i8] c"uDivTest17(42,3) = %u\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @uDivTest1(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @uDivTest2(i32 noundef returned %0, i32 noundef %1) local_unnamed_addr #0 {
  ret i32 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @uDivTest3(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, -1
  %4 = zext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @uDivTest4(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, %1
  %4 = select i1 %3, i32 0, i32 %1
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @uDivTest5(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, %1
  %4 = select i1 %3, i32 0, i32 %1
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @uDivTest6(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, %1
  %4 = select i1 %3, i32 0, i32 %1
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @uDivTest7(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, %1
  %4 = select i1 %3, i32 0, i32 %1
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 536870912) i32 @uDivTest8(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 %0, 3
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 536870912) i32 @uDivTest9(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 %0, 3
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 1073741824) i32 @uDivTest10(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 %0, 2
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 1073741824) i32 @uDivTest11(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 %0, 2
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @uDivTest12(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add i32 %1, 2
  %4 = lshr i32 %0, %3
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @uDivTest13(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add i32 %1, 2
  %4 = lshr i32 %0, %3
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 1073741824) i32 @uDivTest14(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 %0, 2
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 1073741824) i32 @uDivTest15(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 %0, 2
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -2147483648) i32 @uDivTest16(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = sub i32 0, %0
  %4 = lshr i32 %3, 1
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -2147483648) i32 @uDivTest17(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = sub i32 0, %0
  %4 = lshr i32 %3, 1
  ret i32 %4
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 42)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 0)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef 3)
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef 3)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef 3)
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef 3)
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef 5)
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef 5)
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef 10)
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef 10)
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef 1)
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef 1)
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef 10)
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef 10)
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i32 noundef 2147483627)
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16, i32 noundef 2147483627)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
