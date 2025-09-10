; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr66187.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr66187.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 1, align 4
@e = dso_local local_unnamed_addr global i32 -1, align 4
@f = dso_local local_unnamed_addr global i16 0, align 4
@b = dso_local local_unnamed_addr global i16 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @e, align 4, !tbaa !6
  %2 = trunc i32 %1 to i16
  store i16 %2, ptr @f, align 4, !tbaa !10
  %3 = load i16, ptr @b, align 4, !tbaa !10
  %4 = sext i16 %3 to i32
  %5 = icmp sgt i16 %3, -1
  %6 = shl i32 %1, 16
  %7 = ashr exact i32 %6, 16
  %8 = add nsw i32 %7, %4
  %9 = icmp slt i32 %8, 0
  %10 = select i1 %5, i1 %9, i1 false
  br i1 %10, label %11, label %12

11:                                               ; preds = %0
  store i32 0, ptr @a, align 4, !tbaa !6
  br label %16

12:                                               ; preds = %0
  %13 = load i32, ptr @a, align 4, !tbaa !6
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %16, label %15

15:                                               ; preds = %12
  tail call void @abort() #2
  unreachable

16:                                               ; preds = %11, %12
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"short", !8, i64 0}
