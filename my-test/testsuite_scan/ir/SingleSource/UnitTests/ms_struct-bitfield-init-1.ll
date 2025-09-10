; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/ms_struct-bitfield-init-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/ms_struct-bitfield-init-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.anon.0 = type { i8, i8, i8, i8 }
%struct.anon.1 = type { i8, i8, i8, i8 }
%struct.anon.2 = type { i8, i8, i8, i8 }
%struct.anon.3 = type { i8, i8, i8, i8 }
%struct.anon.4 = type { i8, [3 x i8], i8, [3 x i8] }
%struct.anon.5 = type { i8, i8, i8, i8 }
%struct.anon.6 = type { i8, i8 }
%struct.anon.7 = type { i8, [3 x i8] }

@t1 = dso_local local_unnamed_addr global { i8, i8, [2 x i8], i8, i8, [2 x i8] } { i8 1, i8 0, [2 x i8] zeroinitializer, i8 2, i8 0, [2 x i8] zeroinitializer }, align 4
@t2 = dso_local local_unnamed_addr global %struct.anon.0 { i8 3, i8 0, i8 4, i8 0 }, align 4
@t3 = dso_local local_unnamed_addr global %struct.anon.1 { i8 5, i8 0, i8 6, i8 0 }, align 4
@t5 = dso_local local_unnamed_addr global %struct.anon.2 { i8 7, i8 0, i8 8, i8 0 }, align 4
@t6 = dso_local local_unnamed_addr global %struct.anon.3 { i8 5, i8 0, i8 10, i8 0 }, align 4
@t7 = dso_local local_unnamed_addr global %struct.anon.4 { i8 5, [3 x i8] zeroinitializer, i8 10, [3 x i8] zeroinitializer }, align 4
@t8 = dso_local local_unnamed_addr global %struct.anon.5 { i8 5, i8 0, i8 10, i8 0 }, align 4
@t10 = dso_local local_unnamed_addr global %struct.anon.6 { i8 5, i8 10 }, align 4
@t4 = dso_local local_unnamed_addr global %struct.anon.7 zeroinitializer, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @t1, align 4
  %2 = and i32 %1, 4095
  %3 = icmp eq i32 %2, 1
  br i1 %3, label %4, label %8

4:                                                ; preds = %0
  %5 = load i32, ptr getelementptr inbounds nuw (i8, ptr @t1, i64 4), align 4
  %6 = and i32 %5, 4095
  %7 = icmp eq i32 %6, 2
  br i1 %7, label %9, label %8

8:                                                ; preds = %4, %0
  tail call void @abort() #2
  unreachable

9:                                                ; preds = %4
  %10 = load i8, ptr @t2, align 4
  %11 = and i8 %10, 15
  %12 = icmp ne i8 %11, 3
  %13 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t2, i64 2), align 2
  %14 = icmp ne i8 %13, 4
  %15 = select i1 %12, i1 true, i1 %14
  br i1 %15, label %16, label %17

16:                                               ; preds = %9
  tail call void @abort() #2
  unreachable

17:                                               ; preds = %9
  %18 = load i8, ptr @t3, align 4
  %19 = and i8 %18, 15
  %20 = icmp ne i8 %19, 5
  %21 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t3, i64 2), align 2
  %22 = icmp ne i8 %21, 6
  %23 = select i1 %20, i1 true, i1 %22
  br i1 %23, label %24, label %25

24:                                               ; preds = %17
  tail call void @abort() #2
  unreachable

25:                                               ; preds = %17
  %26 = load i8, ptr @t5, align 4
  %27 = and i8 %26, 15
  %28 = icmp ne i8 %27, 7
  %29 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t5, i64 2), align 2
  %30 = icmp ne i8 %29, 8
  %31 = select i1 %28, i1 true, i1 %30
  br i1 %31, label %32, label %33

32:                                               ; preds = %25
  tail call void @abort() #2
  unreachable

33:                                               ; preds = %25
  %34 = load i8, ptr @t6, align 4
  %35 = and i8 %34, 15
  %36 = icmp ne i8 %35, 5
  %37 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t6, i64 2), align 2
  %38 = icmp ne i8 %37, 10
  %39 = select i1 %36, i1 true, i1 %38
  br i1 %39, label %40, label %41

40:                                               ; preds = %33
  tail call void @abort() #2
  unreachable

41:                                               ; preds = %33
  %42 = load i8, ptr @t7, align 4
  %43 = and i8 %42, 15
  %44 = icmp ne i8 %43, 5
  %45 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t7, i64 4), align 4
  %46 = icmp ne i8 %45, 10
  %47 = select i1 %44, i1 true, i1 %46
  br i1 %47, label %48, label %49

48:                                               ; preds = %41
  tail call void @abort() #2
  unreachable

49:                                               ; preds = %41
  %50 = load i8, ptr @t8, align 4
  %51 = and i8 %50, 15
  %52 = icmp eq i8 %51, 5
  br i1 %52, label %53, label %57

53:                                               ; preds = %49
  %54 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t8, i64 2), align 2
  %55 = and i8 %54, 127
  %56 = icmp eq i8 %55, 10
  br i1 %56, label %58, label %57

57:                                               ; preds = %53, %49
  tail call void @abort() #2
  unreachable

58:                                               ; preds = %53
  %59 = load i8, ptr @t10, align 4
  %60 = and i8 %59, 15
  %61 = icmp ne i8 %60, 5
  %62 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t10, i64 1), align 1
  %63 = icmp ne i8 %62, 10
  %64 = select i1 %61, i1 true, i1 %63
  br i1 %64, label %65, label %66

65:                                               ; preds = %58
  tail call void @abort() #2
  unreachable

66:                                               ; preds = %58
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
