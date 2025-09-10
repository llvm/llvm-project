; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr79737-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr79737-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { [9 x i8] }

@i = dso_local local_unnamed_addr global %struct.S zeroinitializer, align 16
@j = dso_local local_unnamed_addr global %struct.S zeroinitializer, align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  store i72 1441160676848959493, ptr @i, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @bar() local_unnamed_addr #0 {
  store i72 1441160676848959493, ptr @j, align 16
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store i72 1441160676848959493, ptr @i, align 16
  store i72 1441160676848959493, ptr @j, align 16
  tail call void asm sideeffect "", "~{memory}"() #3, !srcloc !6
  %1 = load i72, ptr @i, align 16
  %2 = shl i72 %1, 54
  %3 = ashr exact i72 %2, 54
  %4 = trunc nsw i72 %3 to i32
  %5 = load i72, ptr @j, align 16
  %6 = shl i72 %5, 54
  %7 = ashr exact i72 %6, 54
  %8 = trunc nsw i72 %7 to i32
  %9 = icmp eq i32 %4, %8
  br i1 %9, label %10, label %40

10:                                               ; preds = %0
  %11 = shl i72 %1, 53
  %12 = ashr i72 %11, 71
  %13 = trunc nsw i72 %12 to i32
  %14 = shl i72 %5, 53
  %15 = ashr i72 %14, 71
  %16 = trunc nsw i72 %15 to i32
  %17 = icmp eq i32 %13, %16
  br i1 %17, label %18, label %40

18:                                               ; preds = %10
  %19 = shl i72 %1, 29
  %20 = ashr i72 %19, 48
  %21 = trunc nsw i72 %20 to i32
  %22 = shl i72 %5, 29
  %23 = ashr i72 %22, 48
  %24 = trunc nsw i72 %23 to i32
  %25 = icmp eq i32 %21, %24
  br i1 %25, label %26, label %40

26:                                               ; preds = %18
  %27 = shl i72 %1, 14
  %28 = ashr i72 %27, 57
  %29 = trunc nsw i72 %28 to i32
  %30 = shl i72 %5, 14
  %31 = ashr i72 %30, 57
  %32 = trunc nsw i72 %31 to i32
  %33 = icmp eq i32 %29, %32
  br i1 %33, label %34, label %40

34:                                               ; preds = %26
  %35 = ashr i72 %1, 58
  %36 = trunc nsw i72 %35 to i32
  %37 = ashr i72 %5, 58
  %38 = trunc nsw i72 %37 to i32
  %39 = icmp eq i32 %36, %38
  br i1 %39, label %41, label %40

40:                                               ; preds = %34, %26, %18, %10, %0
  tail call void @abort() #4
  unreachable

41:                                               ; preds = %34
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 455}
