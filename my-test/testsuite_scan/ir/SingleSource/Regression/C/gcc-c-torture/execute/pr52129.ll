; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr52129.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr52129.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.T = type { [64 x i8], [64 x i8] }

@t = dso_local global %struct.T zeroinitializer, align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef i32 @foo(ptr noundef readnone captures(address) %0, [2 x i64] %1, ptr noundef readnone captures(address) %2, ptr noundef readnone captures(address) %3) local_unnamed_addr #0 {
  %5 = extractvalue [2 x i64] %1, 0
  %6 = extractvalue [2 x i64] %1, 1
  %7 = icmp ne ptr %0, getelementptr inbounds nuw (i8, ptr @t, i64 2)
  %8 = icmp ne i64 %5, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @t, i64 69) to i64)
  %9 = select i1 %7, i1 true, i1 %8
  %10 = and i64 %6, 4294967295
  %11 = icmp ne i64 %10, 27
  %12 = select i1 %9, i1 true, i1 %11
  %13 = icmp ne ptr %2, getelementptr inbounds nuw (i8, ptr @t, i64 17)
  %14 = select i1 %12, i1 true, i1 %13
  %15 = icmp ne ptr %3, getelementptr inbounds nuw (i8, ptr @t, i64 81)
  %16 = select i1 %14, i1 true, i1 %15
  br i1 %16, label %17, label %18

17:                                               ; preds = %4
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %4
  ret i32 29
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef i32 @bar(ptr noundef readnone captures(address) %0, ptr readnone captures(none) %1, ptr readnone captures(none) %2, [2 x i64] %3, i32 noundef %4, ptr noundef readnone captures(address) %5) local_unnamed_addr #0 {
  %7 = sext i32 %4 to i64
  %8 = getelementptr inbounds i8, ptr %5, i64 %7
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 64
  %10 = getelementptr inbounds i8, ptr %9, i64 %7
  %11 = tail call i32 @foo(ptr noundef %0, [2 x i64] %3, ptr noundef %8, ptr noundef nonnull %10)
  ret i32 29
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = tail call i32 @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @t, i64 2), ptr poison, ptr poison, [2 x i64] [i64 ptrtoint (ptr getelementptr (i8, ptr @t, i64 69) to i64), i64 27], i32 noundef 17, ptr noundef nonnull @t)
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
