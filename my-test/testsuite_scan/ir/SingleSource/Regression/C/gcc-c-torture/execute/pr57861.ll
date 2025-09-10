; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57861.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57861.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i16 1, align 4
@b = dso_local global i32 0, align 4
@g = dso_local local_unnamed_addr global ptr @b, align 8
@f = dso_local local_unnamed_addr global i16 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@i = dso_local local_unnamed_addr global i32 0, align 4
@j = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i16, ptr @a, align 4, !tbaa !6
  %2 = trunc i16 %1 to i8
  store i32 0, ptr @j, align 4, !tbaa !10
  %3 = and i16 %1, 255
  %4 = zext nneg i16 %3 to i32
  %5 = icmp ne i8 %2, 0
  %6 = zext i1 %5 to i32
  store i32 %6, ptr @i, align 4, !tbaa !10
  %7 = load i32, ptr @e, align 4, !tbaa !10
  %8 = icmp ult i32 %7, %4
  br i1 %8, label %9, label %16

9:                                                ; preds = %0
  %10 = load i32, ptr @d, align 4
  %11 = icmp ne i32 %10, 0
  %12 = load i32, ptr @h, align 4
  %13 = icmp ne i32 %12, 0
  %14 = select i1 %11, i1 %13, i1 false
  %15 = zext i1 %14 to i16
  store i16 %15, ptr @a, align 4, !tbaa !6
  store i16 0, ptr @f, align 4, !tbaa !6
  br label %16

16:                                               ; preds = %0, %9
  %17 = phi i16 [ %15, %9 ], [ %1, %0 ]
  store i32 2, ptr @c, align 4, !tbaa !10
  %18 = load ptr, ptr @g, align 8, !tbaa !12
  store i32 0, ptr %18, align 4, !tbaa !10
  %19 = icmp eq i16 %17, 0
  br i1 %19, label %21, label %20

20:                                               ; preds = %16
  tail call void @abort() #2
  unreachable

21:                                               ; preds = %16
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
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"p1 int", !14, i64 0}
!14 = !{!"any pointer", !8, i64 0}
