; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58277-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58277-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@n = dso_local local_unnamed_addr global i8 0, align 4
@d = dso_local global i32 0, align 4
@r = dso_local global ptr null, align 8
@f = dso_local global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@o = dso_local local_unnamed_addr global i32 0, align 4
@x = dso_local local_unnamed_addr global i32 0, align 4
@h = internal global ptr @f, align 8
@s = internal global ptr @r, align 8

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(readwrite, argmem: write) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @g, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %5, label %3

3:                                                ; preds = %0
  %4 = load volatile i32, ptr @d, align 4, !tbaa !6
  br label %5

5:                                                ; preds = %3, %0
  %6 = load volatile ptr, ptr @h, align 8, !tbaa !10
  store i32 0, ptr %6, align 4, !tbaa !6
  %7 = load volatile ptr, ptr @s, align 8, !tbaa !13
  store ptr null, ptr %7, align 8, !tbaa !17
  store i8 0, ptr @n, align 4, !tbaa !19
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nounwind willreturn memory(readwrite, argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!11 = !{!"p1 int", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"p3 int", !15, i64 0}
!15 = !{!"any p3 pointer", !16, i64 0}
!16 = !{!"any p2 pointer", !12, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"p2 int", !16, i64 0}
!19 = !{!8, !8, i64 0}
