; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/testcase-InstCombine-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/testcase-InstCombine-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { [9 x i8] }

@f = dso_local local_unnamed_addr global i32 0, align 4
@__const.main.m = private unnamed_addr constant { i8, i8, i8, i8, i8, i8, i8, i8, i8 } { i8 5, i8 0, i8 -40, i8 -1, i8 -1, i8 79, i8 0, i8 20, i8 0 }, align 4
@h = dso_local local_unnamed_addr global %struct.S zeroinitializer, align 16

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @f, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %8, label %3

3:                                                ; preds = %0, %3
  %4 = phi i32 [ %5, %3 ], [ %1, %0 ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(9) @h, ptr noundef nonnull align 4 dereferenceable(9) @__const.main.m, i64 9, i1 false), !tbaa.struct !10
  %5 = add nsw i32 %4, 1
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %3, !llvm.loop !12

7:                                                ; preds = %3
  store i32 0, ptr @f, align 4, !tbaa !6
  br label %18

8:                                                ; preds = %0
  %9 = load i72, ptr @h, align 16
  %10 = shl i72 %9, 14
  %11 = ashr i72 %10, 57
  %12 = trunc nsw i72 %11 to i32
  %13 = trunc nsw i72 %11 to i64
  %14 = mul nsw i32 %12, %12
  %15 = zext nneg i32 %14 to i64
  %16 = sub nsw i64 0, %15
  %17 = icmp eq i64 %13, %16
  br i1 %17, label %20, label %18

18:                                               ; preds = %7, %8
  %19 = tail call i32 @putchar(i32 49)
  br label %20

20:                                               ; preds = %18, %8
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind }

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
!10 = !{i64 0, i64 9, !11}
!11 = !{!8, !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
