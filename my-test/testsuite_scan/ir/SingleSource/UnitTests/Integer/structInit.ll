; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/structInit.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/structInit.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@link = dso_local global [3 x { i32, [10 x i8], [2 x i8], i32 }] [{ i32, [10 x i8], [2 x i8], i32 } { i32 1, [10 x i8] c"link1\00\00\00\00\00", [2 x i8] zeroinitializer, i32 10 }, { i32, [10 x i8], [2 x i8], i32 } { i32 2, [10 x i8] c"link2\00\00\00\00\00", [2 x i8] zeroinitializer, i32 20 }, { i32, [10 x i8], [2 x i8], i32 } { i32 3, [10 x i8] c"link3\00\00\00\00\00", [2 x i8] zeroinitializer, i32 30 }], align 4
@.str = private unnamed_addr constant [12 x i8] c"%d, %s, %d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @link, align 4, !tbaa !6
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @link, i64 16), align 4, !tbaa !11
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %1, ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @link, i64 4), i32 noundef %2)
  %4 = load i32, ptr getelementptr inbounds nuw (i8, ptr @link, i64 20), align 4, !tbaa !6
  %5 = load i32, ptr getelementptr inbounds nuw (i8, ptr @link, i64 36), align 4, !tbaa !11
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %4, ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @link, i64 24), i32 noundef %5)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @link, i64 40), align 4, !tbaa !6
  %8 = load i32, ptr getelementptr inbounds nuw (i8, ptr @link, i64 56), align 4, !tbaa !11
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %7, ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @link, i64 44), i32 noundef %8)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"Connection_Type", !8, i64 0, !9, i64 4, !8, i64 16}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !8, i64 16}
