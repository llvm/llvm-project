; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/shr.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/shr.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [35 x i8] c"ShrTest1(-1, 11, 0xFF000000) = %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [27 x i8] c"ShrTest2(-73, 11, 1) = %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [20 x i8] c"ShrTest3(-73) = %d\0A\00", align 1
@.str.3 = private unnamed_addr constant [20 x i8] c"ShrTest4(-73) = %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [25 x i8] c"ShrTest5(7773, 11) = %d\0A\00", align 1
@.str.5 = private unnamed_addr constant [24 x i8] c"ShrTest6(-73, 11) = %d\0A\00", align 1
@.str.6 = private unnamed_addr constant [21 x i8] c"ShrTest7(-737) = %d\0A\00", align 1
@.str.7 = private unnamed_addr constant [19 x i8] c"ShrTest8(73) = %d\0A\00", align 1
@.str.8 = private unnamed_addr constant [29 x i8] c"ShrTest9(13, 7773, 11) = %d\0A\00", align 1
@.str.9 = private unnamed_addr constant [30 x i8] c"ShrTest10(13, 7773, 11) = %d\0A\00", align 1
@.str.10 = private unnamed_addr constant [30 x i8] c"ShrTest11(13, 7773, 11) = %d\0A\00", align 1
@.str.11 = private unnamed_addr constant [37 x i8] c"ShrTest12(7773, -73, 11, 3773) = %d\0A\00", align 1
@.str.12 = private unnamed_addr constant [37 x i8] c"ShrTest13(7773, -73, 11, 3773) = %d\0A\00", align 1
@.str.13 = private unnamed_addr constant [37 x i8] c"ShrTest14(7773, -73, 11, 3773) = %d\0A\00", align 1
@.str.14 = private unnamed_addr constant [30 x i8] c"ShrTest15(7773, 13, 11) = %d\0A\00", align 1
@.str.15 = private unnamed_addr constant [29 x i8] c"ShrTest16(-73, 11, 13) = %d\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest1(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = lshr i32 %0, %1
  %5 = and i32 %4, %2
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest2(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = ashr i32 %0, %1
  %5 = and i32 %4, %2
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -1, 1) i32 @ShrTest3(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 31
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 2) i32 @ShrTest4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 31
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest5(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 %0, %1
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest6(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 -1, %1
  %4 = ashr i32 %0, %1
  %5 = and i32 %4, %3
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @ShrTest7(i32 noundef returned %0) local_unnamed_addr #0 {
  ret i32 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @ShrTest8(i32 noundef %0) local_unnamed_addr #0 {
  ret i32 -1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest9(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = ashr i32 %1, %2
  %5 = add nsw i32 %4, %0
  %6 = shl i32 %5, %2
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest10(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = ashr i32 %1, %2
  %5 = and i32 %4, %0
  %6 = shl i32 %5, %2
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest11(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = ashr i32 %1, %2
  %5 = or i32 %4, %0
  %6 = shl i32 %5, %2
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest12(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = ashr i32 %1, %2
  %6 = and i32 %5, %3
  %7 = add nsw i32 %6, %0
  %8 = shl i32 %7, %2
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest13(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = ashr i32 %1, %2
  %6 = and i32 %3, %0
  %7 = and i32 %6, %5
  %8 = shl i32 %7, %2
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest14(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = ashr i32 %1, %2
  %6 = and i32 %5, %3
  %7 = or i32 %6, %0
  %8 = shl i32 %7, %2
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest15(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = shl i32 %0, %1
  %5 = lshr i32 %4, %2
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @ShrTest16(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = ashr i32 %0, %1
  %5 = shl i32 %4, %2
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 1)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef -1)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef 1)
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef 3)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef 2097151)
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef -737)
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef -1)
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef 32768)
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef 2048)
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef 30720)
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef 23646208)
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef 7399424)
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef 16246784)
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef 31092)
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i32 noundef -8192)
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
