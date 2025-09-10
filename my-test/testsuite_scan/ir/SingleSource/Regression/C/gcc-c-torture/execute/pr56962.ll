; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr56962.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr56962.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v = dso_local global [144 x i64] zeroinitializer, align 8

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @bar(ptr noundef readnone captures(address) %0) local_unnamed_addr #0 {
  %2 = icmp eq ptr %0, getelementptr inbounds nuw (i8, ptr @v, i64 232)
  br i1 %2, label %4, label %3

3:                                                ; preds = %1
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo(ptr noundef captures(address) %0, i64 noundef %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = mul nsw i64 %1, 3
  %5 = shl i64 %2, 5
  %6 = getelementptr i8, ptr %0, i64 %5
  %7 = getelementptr i64, ptr %6, i64 %4
  %8 = load i64, ptr %7, align 8, !tbaa !6
  %9 = mul i64 %2, 40
  %10 = getelementptr i8, ptr %0, i64 %9
  %11 = getelementptr i64, ptr %10, i64 %4
  %12 = load i64, ptr %11, align 8, !tbaa !6
  %13 = shl nsw i64 %1, 2
  %14 = getelementptr i64, ptr %10, i64 %13
  %15 = load i64, ptr %14, align 8, !tbaa !6
  %16 = getelementptr inbounds i64, ptr %0, i64 %13
  store i64 %8, ptr %16, align 8, !tbaa !6
  %17 = getelementptr i64, ptr %10, i64 %1
  tail call void @bar(ptr noundef %17)
  %18 = add nsw i64 %15, %12
  %19 = add i64 %2, %1
  %20 = mul i64 %19, 40
  %21 = getelementptr inbounds i8, ptr %0, i64 %20
  store i64 %18, ptr %21, align 8, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  tail call void @foo(ptr noundef nonnull @v, i64 noundef 24, i64 noundef 1)
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
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
