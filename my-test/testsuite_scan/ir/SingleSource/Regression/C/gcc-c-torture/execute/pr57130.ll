; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57130.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57130.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { i32, i32, i32, i32 }

@s = dso_local global [2 x %struct.S] [%struct.S { i32 6, i32 8, i32 -8, i32 -5 }, %struct.S { i32 0, i32 2, i32 -1, i32 2 }], align 4
@foo.cnt = internal unnamed_addr global i32 0, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo([2 x i64] %0) local_unnamed_addr #0 {
  %2 = alloca %struct.S, align 8
  %3 = extractvalue [2 x i64] %0, 0
  store i64 %3, ptr %2, align 8
  %4 = extractvalue [2 x i64] %0, 1
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i64 %4, ptr %5, align 8
  %6 = load i32, ptr @foo.cnt, align 4, !tbaa !6
  %7 = add nsw i32 %6, 1
  store i32 %7, ptr @foo.cnt, align 4, !tbaa !6
  %8 = sext i32 %6 to i64
  %9 = getelementptr inbounds %struct.S, ptr @s, i64 %8
  %10 = call i32 @bcmp(ptr noundef nonnull dereferenceable(16) %2, ptr noundef nonnull dereferenceable(16) %9, i64 16)
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %13, label %12

12:                                               ; preds = %1
  tail call void @abort() #4
  unreachable

13:                                               ; preds = %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  tail call void @foo([2 x i64] [i64 34359738374, i64 -17179869192])
  tail call void @foo([2 x i64] [i64 8589934592, i64 12884901887])
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #3

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { noreturn nounwind }

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
