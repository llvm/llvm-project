; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr45034.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr45034.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo(i32 %0, i32 noundef %1, i32 %2) local_unnamed_addr #0 {
  %4 = add i32 %1, -128
  %5 = icmp ult i32 %4, -256
  br i1 %5, label %6, label %7

6:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

7:                                                ; preds = %3
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @test_neg() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i32 [ %6, %1 ], [ -128, %0 ]
  %3 = mul i32 %2, -16777216
  %4 = ashr exact i32 %3, 24
  tail call void @foo(i32 poison, i32 noundef %4, i32 poison)
  %5 = icmp eq i32 %2, 127
  %6 = add nsw i32 %2, 1
  br i1 %5, label %7, label %1

7:                                                ; preds = %1
  ret i32 0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i32 [ %6, %1 ], [ -128, %0 ]
  %3 = mul i32 %2, -16777216
  %4 = ashr exact i32 %3, 24
  tail call void @foo(i32 poison, i32 noundef %4, i32 poison)
  %5 = icmp eq i32 %2, 127
  %6 = add nsw i32 %2, 1
  br i1 %5, label %7, label %1

7:                                                ; preds = %1
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
