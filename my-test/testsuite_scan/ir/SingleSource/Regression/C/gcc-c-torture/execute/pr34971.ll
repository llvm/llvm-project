; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr34971.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr34971.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.foo = type { i64 }

@x = dso_local local_unnamed_addr global %struct.foo zeroinitializer, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @test1(i64 noundef %0) local_unnamed_addr #0 {
  %2 = load i64, ptr @x, align 8
  %3 = and i64 %2, 1099511627775
  %4 = shl nuw nsw i64 %3, 8
  %5 = lshr i64 %3, 32
  %6 = or disjoint i64 %4, %5
  %7 = icmp eq i64 %6, %0
  br i1 %7, label %9, label %8

8:                                                ; preds = %1
  tail call void @abort() #3
  unreachable

9:                                                ; preds = %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: cold nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = load i64, ptr @x, align 8
  %2 = and i64 %1, -1099511627776
  %3 = or disjoint i64 %2, 4294967297
  store i64 %3, ptr @x, align 8
  tail call void @abort() #3
  unreachable
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
