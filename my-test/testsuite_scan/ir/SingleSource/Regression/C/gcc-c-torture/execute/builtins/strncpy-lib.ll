; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strncpy-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strncpy-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef ptr @strncpy(ptr noundef returned writeonly captures(ret: address, provenance) %0, ptr noundef readonly captures(none) %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = load i32, ptr @inside_main, align 4, !tbaa !6
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %11

6:                                                ; preds = %3
  %7 = load i8, ptr %1, align 1, !tbaa !10
  %8 = icmp ne i8 %7, 0
  %9 = icmp ne i64 %2, 0
  %10 = and i1 %8, %9
  br i1 %10, label %17, label %12

11:                                               ; preds = %3
  tail call void @abort() #3
  unreachable

12:                                               ; preds = %17, %6
  %13 = phi i64 [ %2, %6 ], [ %24, %17 ]
  %14 = phi ptr [ %0, %6 ], [ %23, %17 ]
  %15 = icmp eq i64 %13, 0
  br i1 %15, label %29, label %16

16:                                               ; preds = %12
  tail call void @llvm.memset.p0.i64(ptr align 1 %14, i8 0, i64 %13, i1 false), !tbaa !10
  br label %29

17:                                               ; preds = %6, %17
  %18 = phi i8 [ %25, %17 ], [ %7, %6 ]
  %19 = phi ptr [ %23, %17 ], [ %0, %6 ]
  %20 = phi i64 [ %24, %17 ], [ %2, %6 ]
  %21 = phi ptr [ %22, %17 ], [ %1, %6 ]
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 1
  %23 = getelementptr inbounds nuw i8, ptr %19, i64 1
  store i8 %18, ptr %19, align 1, !tbaa !10
  %24 = add i64 %20, -1
  %25 = load i8, ptr %22, align 1, !tbaa !10
  %26 = icmp ne i8 %25, 0
  %27 = icmp ne i64 %24, 0
  %28 = select i1 %26, i1 %27, i1 false
  br i1 %28, label %17, label %12, !llvm.loop !11

29:                                               ; preds = %16, %12
  ret ptr %0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: write) }
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
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
