; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr40022.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr40022.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { ptr }

@g = dso_local global %struct.A zeroinitializer, align 8
@f = dso_local global %struct.A zeroinitializer, align 8
@d = dso_local global %struct.B zeroinitializer, align 8
@e = dso_local global %struct.A zeroinitializer, align 8

; Function Attrs: noinline nounwind uwtable
define dso_local noundef ptr @foo(ptr noundef returned %0) local_unnamed_addr #0 {
  tail call void asm sideeffect "", "imr,~{memory}"(ptr %0) #3, !srcloc !6
  ret ptr %0
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @bar(ptr noundef writeonly captures(none) initializes((0, 8)) %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #0 {
  %5 = tail call ptr @foo(ptr noundef %1)
  store ptr %1, ptr %0, align 8, !tbaa !7
  %6 = icmp eq ptr %1, null
  br i1 %6, label %13, label %7, !llvm.loop !12

7:                                                ; preds = %4
  br label %8, !llvm.loop !12

8:                                                ; preds = %7, %8
  %9 = phi ptr [ %1, %7 ], [ %10, %8 ]
  %10 = load ptr, ptr %9, align 8, !tbaa !7
  %11 = icmp eq ptr %10, null
  br i1 %11, label %12, label %8, !llvm.loop !12

12:                                               ; preds = %8
  br label %13, !llvm.loop !12

13:                                               ; preds = %12, %4
  %14 = phi ptr [ %9, %12 ], [ %0, %4 ]
  %15 = tail call ptr @foo(ptr noundef %2)
  store ptr %2, ptr %14, align 8, !tbaa !7
  %16 = icmp eq ptr %2, null
  br i1 %16, label %23, label %17, !llvm.loop !14

17:                                               ; preds = %13
  br label %18, !llvm.loop !14

18:                                               ; preds = %17, %18
  %19 = phi ptr [ %2, %17 ], [ %20, %18 ]
  %20 = load ptr, ptr %19, align 8, !tbaa !7
  %21 = icmp eq ptr %20, null
  br i1 %21, label %22, label %18, !llvm.loop !14

22:                                               ; preds = %18
  br label %23, !llvm.loop !14

23:                                               ; preds = %22, %13
  %24 = phi ptr [ %19, %22 ], [ %14, %13 ]
  %25 = tail call ptr @foo(ptr noundef %3)
  store ptr %3, ptr %24, align 8, !tbaa !7
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store ptr @g, ptr @f, align 8, !tbaa !15
  tail call void @bar(ptr noundef nonnull @d, ptr noundef nonnull @e, ptr noundef nonnull @f, ptr noundef null)
  %1 = load ptr, ptr @d, align 8, !tbaa !17
  %2 = icmp eq ptr %1, null
  br i1 %2, label %12, label %3

3:                                                ; preds = %0
  %4 = load ptr, ptr %1, align 8, !tbaa !15
  %5 = icmp eq ptr %4, null
  br i1 %5, label %12, label %6

6:                                                ; preds = %3
  %7 = load ptr, ptr %4, align 8, !tbaa !15
  %8 = icmp eq ptr %7, null
  br i1 %8, label %12, label %9

9:                                                ; preds = %6
  %10 = load ptr, ptr %7, align 8, !tbaa !15
  %11 = icmp eq ptr %10, null
  br i1 %11, label %13, label %12

12:                                               ; preds = %9, %6, %3, %0
  tail call void @abort() #4
  unreachable

13:                                               ; preds = %9
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 162}
!7 = !{!8, !8, i64 0}
!8 = !{!"p1 _ZTS1A", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = !{!16, !8, i64 0}
!16 = !{!"A", !8, i64 0}
!17 = !{!18, !8, i64 0}
!18 = !{!"B", !8, i64 0}
