; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr70127.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr70127.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { i32, i8 }

@c = dso_local local_unnamed_addr global { i32, i8, [3 x i8] } { i32 5, i8 1, [3 x i8] zeroinitializer }, align 8
@b = dso_local local_unnamed_addr global i16 0, align 4
@d = dso_local local_unnamed_addr global %struct.S zeroinitializer, align 8
@a = dso_local local_unnamed_addr global [1 x %struct.S] zeroinitializer, align 8

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 1
  br i1 %2, label %4, label %3

3:                                                ; preds = %1
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = load i16, ptr @b, align 4, !tbaa !6
  %2 = icmp slt i16 %1, 1
  br i1 %2, label %6, label %3

3:                                                ; preds = %0
  %4 = add nuw i16 %1, 1
  %5 = load i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 4), align 4
  br label %10

6:                                                ; preds = %0
  %7 = load i64, ptr @c, align 8
  store i64 %7, ptr @a, align 8
  store i64 %7, ptr @d, align 8
  %8 = lshr i64 %7, 32
  %9 = trunc i64 %8 to i8
  br label %10

10:                                               ; preds = %3, %6
  %11 = phi i8 [ %5, %3 ], [ %9, %6 ]
  %12 = phi i16 [ %4, %3 ], [ 2, %6 ]
  store i16 %12, ptr @b, align 4, !tbaa !6
  %13 = shl i8 %11, 6
  %14 = ashr exact i8 %13, 6
  %15 = sext i8 %14 to i32
  tail call void @foo(i32 noundef %15)
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
