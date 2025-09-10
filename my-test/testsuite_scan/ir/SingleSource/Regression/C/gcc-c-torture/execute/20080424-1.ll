; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20080424-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20080424-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@bar.i = internal unnamed_addr global i32 0, align 4
@g = dso_local global [48 x [3 x [3 x i32]]] zeroinitializer, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @bar(ptr noundef readnone captures(address) %0, ptr noundef readnone captures(address) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @bar.i, align 4, !tbaa !6
  %4 = sext i32 %3 to i64
  %5 = getelementptr [3 x [3 x i32]], ptr @g, i64 %4
  %6 = getelementptr i8, ptr %5, i64 288
  %7 = icmp eq ptr %0, %6
  br i1 %7, label %8, label %11

8:                                                ; preds = %2
  %9 = add nsw i32 %3, 1
  store i32 %9, ptr @bar.i, align 4, !tbaa !6
  %10 = icmp eq ptr %1, %5
  br i1 %10, label %12, label %11

11:                                               ; preds = %8, %2
  tail call void @abort() #3
  unreachable

12:                                               ; preds = %8
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  tail call void @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 288), ptr noundef nonnull @g)
  tail call void @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 324), ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 36))
  tail call void @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 360), ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 72))
  tail call void @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 396), ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 108))
  tail call void @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 432), ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 144))
  tail call void @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 468), ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 180))
  tail call void @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 504), ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 216))
  tail call void @bar(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 540), ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @g, i64 252))
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
