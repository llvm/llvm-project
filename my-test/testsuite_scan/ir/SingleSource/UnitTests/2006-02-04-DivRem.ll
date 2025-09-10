; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2006-02-04-DivRem.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2006-02-04-DivRem.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"%u %u %u %u\0A\00", align 1
@i = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local void @test(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = and i32 %1, 15
  %4 = shl nuw nsw i32 8, %3
  %5 = add nuw nsw i32 %3, 3
  %6 = lshr i32 %0, %5
  %7 = add nsw i32 %4, -1
  %8 = and i32 %7, %0
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %0, i32 noundef %1, i32 noundef %6, i32 noundef %8)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  store i32 10, ptr @i, align 4, !tbaa !6
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i32 [ 10, %0 ], [ %13, %1 ]
  %3 = xor i32 %2, 12345
  %4 = and i32 %3, 15
  %5 = shl nuw nsw i32 8, %4
  %6 = add nuw nsw i32 %4, 3
  %7 = lshr i32 %2, %6
  %8 = add nsw i32 %5, -1
  %9 = and i32 %8, %2
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %2, i32 noundef %3, i32 noundef %7, i32 noundef %9)
  %11 = load i32, ptr @i, align 4, !tbaa !6
  %12 = mul i32 %11, -3
  %13 = add i32 %12, -3
  store i32 %13, ptr @i, align 4, !tbaa !6
  %14 = icmp slt i32 %13, 139045193
  br i1 %14, label %1, label %15, !llvm.loop !10

15:                                               ; preds = %1
  ret i32 0
}

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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
