; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020201-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020201-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@cx = dso_local local_unnamed_addr global i8 7, align 4
@sx = dso_local local_unnamed_addr global i16 14, align 4
@ix = dso_local local_unnamed_addr global i32 21, align 4
@lx = dso_local local_unnamed_addr global i64 28, align 8
@Lx = dso_local local_unnamed_addr global i64 35, align 8

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i8, ptr @cx, align 4, !tbaa !6
  %2 = add i8 %1, -6
  %3 = icmp ult i8 %2, 6
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %0
  %6 = icmp eq i8 %1, 7
  br i1 %6, label %8, label %7

7:                                                ; preds = %5
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %5
  %9 = load i16, ptr @sx, align 4, !tbaa !9
  %10 = add i16 %9, -12
  %11 = icmp ult i16 %10, 6
  br i1 %11, label %13, label %12

12:                                               ; preds = %8
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %8
  %14 = trunc nuw nsw i16 %9 to i8
  %15 = urem i8 %14, 6
  %16 = icmp eq i8 %15, 2
  br i1 %16, label %18, label %17

17:                                               ; preds = %13
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %13
  %19 = load i32, ptr @ix, align 4, !tbaa !11
  %20 = add i32 %19, -18
  %21 = icmp ult i32 %20, 6
  br i1 %21, label %23, label %22

22:                                               ; preds = %18
  tail call void @abort() #3
  unreachable

23:                                               ; preds = %18
  %24 = trunc nuw nsw i32 %19 to i8
  %25 = urem i8 %24, 6
  %26 = icmp eq i8 %25, 3
  br i1 %26, label %28, label %27

27:                                               ; preds = %23
  tail call void @abort() #3
  unreachable

28:                                               ; preds = %23
  %29 = load i64, ptr @lx, align 8, !tbaa !13
  %30 = add i64 %29, -24
  %31 = icmp ult i64 %30, 6
  br i1 %31, label %33, label %32

32:                                               ; preds = %28
  tail call void @abort() #3
  unreachable

33:                                               ; preds = %28
  %34 = trunc nuw nsw i64 %29 to i8
  %35 = urem i8 %34, 6
  %36 = icmp eq i8 %35, 4
  br i1 %36, label %38, label %37

37:                                               ; preds = %33
  tail call void @abort() #3
  unreachable

38:                                               ; preds = %33
  %39 = load i64, ptr @Lx, align 8, !tbaa !15
  %40 = add i64 %39, -30
  %41 = icmp ult i64 %40, 6
  br i1 %41, label %43, label %42

42:                                               ; preds = %38
  tail call void @abort() #3
  unreachable

43:                                               ; preds = %38
  %44 = trunc nuw nsw i64 %39 to i8
  %45 = urem i8 %44, 6
  %46 = icmp eq i8 %45, 5
  br i1 %46, label %48, label %47

47:                                               ; preds = %43
  tail call void @abort() #3
  unreachable

48:                                               ; preds = %43
  tail call void @exit(i32 noundef 0) #3
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"short", !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !7, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"long", !7, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"long long", !7, i64 0}
