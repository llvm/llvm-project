; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/scal-to-vec1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/scal-to-vec1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@one = dso_local global i32 1, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = load volatile i32, ptr @one, align 4, !tbaa !6
  %4 = trunc i32 %3 to i16
  %5 = shl i32 %3, 16
  %6 = add i32 %5, 131072
  %7 = ashr exact i32 %6, 16
  %8 = shl i32 %3, 16
  %9 = ashr exact i32 %8, 16
  %10 = add nsw i32 %9, 2
  %11 = icmp eq i32 %10, %7
  br i1 %11, label %12, label %18

12:                                               ; preds = %2
  %13 = shl i32 %3, 16
  %14 = sub i32 131072, %13
  %15 = ashr exact i32 %14, 16
  %16 = sub nsw i32 2, %9
  %17 = icmp eq i32 %16, %15
  br i1 %17, label %19, label %24

18:                                               ; preds = %2
  tail call void @abort() #2
  unreachable

19:                                               ; preds = %12
  %20 = shl i16 %4, 1
  %21 = sext i16 %20 to i32
  %22 = ashr exact i32 %8, 15
  %23 = icmp eq i32 %22, %21
  br i1 %23, label %26, label %25

24:                                               ; preds = %12
  tail call void @abort() #2
  unreachable

25:                                               ; preds = %19
  tail call void @abort() #2
  unreachable

26:                                               ; preds = %19
  %27 = shl i16 2, %4
  %28 = sext i16 %27 to i32
  %29 = and i32 %3, 65535
  %30 = shl i32 2, %29
  %31 = icmp eq i32 %30, %28
  br i1 %31, label %32, label %37

32:                                               ; preds = %26
  %33 = lshr i16 2, %4
  %34 = zext nneg i16 %33 to i32
  %35 = lshr i32 2, %29
  %36 = icmp eq i32 %35, %34
  br i1 %36, label %39, label %38

37:                                               ; preds = %26
  tail call void @abort() #2
  unreachable

38:                                               ; preds = %32
  tail call void @abort() #2
  unreachable

39:                                               ; preds = %32
  %40 = shl i32 %3, 16
  %41 = add i32 %40, -131072
  %42 = ashr exact i32 %41, 16
  %43 = add nsw i32 %9, -2
  %44 = icmp eq i32 %43, %42
  br i1 %44, label %46, label %45

45:                                               ; preds = %39
  tail call void @abort() #2
  unreachable

46:                                               ; preds = %39
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
