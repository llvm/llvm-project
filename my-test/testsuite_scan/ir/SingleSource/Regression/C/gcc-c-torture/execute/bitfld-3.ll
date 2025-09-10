; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/bitfld-3.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/bitfld-3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global { i8, i8, i8, i8, i8, [3 x i8], i8, i8, i8, i8, i8, [3 x i8], i8, i8, i8, i8, i8, i8, [2 x i8] } { i8 0, i8 0, i8 16, i8 0, i8 0, [3 x i8] zeroinitializer, i8 0, i8 0, i8 16, i8 0, i8 0, [3 x i8] zeroinitializer, i8 0, i8 0, i8 16, i8 0, i8 0, i8 0, [2 x i8] zeroinitializer }, align 8
@b = dso_local local_unnamed_addr global { i8, i8, i8, i8, i8, [3 x i8], i8, i8, i8, i8, i8, [3 x i8], i8, i8, i8, i8, i8, i8, [2 x i8] } { i8 0, i8 0, i8 0, i8 0, i8 1, [3 x i8] zeroinitializer, i8 0, i8 0, i8 0, i8 0, i8 1, [3 x i8] zeroinitializer, i8 0, i8 0, i8 0, i8 0, i8 1, i8 0, [2 x i8] zeroinitializer }, align 8
@c = dso_local local_unnamed_addr global { i8, i8, i8, i8, i8, [3 x i8], i8, i8, i8, i8, i8, [3 x i8], i8, i8, i8, i8, i8, i8, [2 x i8] } { i8 -1, i8 -1, i8 -1, i8 -1, i8 1, [3 x i8] zeroinitializer, i8 0, i8 0, i8 0, i8 0, i8 0, [3 x i8] zeroinitializer, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, [2 x i8] zeroinitializer }, align 8

; Function Attrs: cold nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i64, ptr @a, align 8
  %2 = and i64 %1, 8589934591
  %3 = mul i64 %2, %2
  %4 = icmp eq i64 %3, 0
  br i1 %4, label %5, label %12

5:                                                ; preds = %0
  %6 = load i64, ptr getelementptr inbounds nuw (i8, ptr @a, i64 8), align 8
  %7 = and i64 %6, 1099511627775
  %8 = mul i64 %7, %2
  %9 = mul i64 %7, %7
  %10 = or i64 %8, %9
  %11 = icmp eq i64 %10, 0
  br i1 %11, label %13, label %12

12:                                               ; preds = %5, %0
  tail call void @abort() #2
  unreachable

13:                                               ; preds = %5
  tail call void @abort() #2
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { cold nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
