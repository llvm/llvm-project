; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/testcase-CGExprConstant.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/testcase-CGExprConstant.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S1 = type { i32, i32 }
%struct.S2 = type { %struct.S1, ptr, [2 x i32] }

@gs1 = dso_local global %struct.S1 { i32 1, i32 2 }, align 4
@.compoundliteral = internal global %struct.S2 { %struct.S1 { i32 1, i32 2 }, ptr @gs1, [2 x i32] [i32 1, i32 2] }, align 8
@s = dso_local local_unnamed_addr global ptr @.compoundliteral, align 8
@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local range(i32 0, 7) i32 @main() local_unnamed_addr #0 {
  %1 = load ptr, ptr @s, align 8, !tbaa !6
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %3 = load i32, ptr %2, align 4, !tbaa !11
  %4 = icmp eq i32 %3, 2
  br i1 %4, label %5, label %11

5:                                                ; preds = %0
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %7 = load i32, ptr %6, align 4, !tbaa !16
  %8 = icmp eq i32 %7, 2
  br i1 %8, label %9, label %11

9:                                                ; preds = %5
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 2)
  br label %11

11:                                               ; preds = %5, %0, %9
  %12 = phi i32 [ 0, %9 ], [ 2, %0 ], [ 6, %5 ]
  ret i32 %12
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
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS2S2", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !14, i64 4}
!12 = !{!"S2", !13, i64 0, !15, i64 8, !9, i64 16}
!13 = !{!"S1", !14, i64 0, !14, i64 4}
!14 = !{!"int", !9, i64 0}
!15 = !{!"p1 _ZTS2S1", !8, i64 0}
!16 = !{!14, !14, i64 0}
