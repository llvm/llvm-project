; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/random.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/random.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [6 x i8] c"%.9f\0A\00", align 1
@gen_random.last = internal unnamed_addr global i64 42, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %6, label %4

4:                                                ; preds = %2
  %5 = load i64, ptr @gen_random.last, align 8
  br label %14

6:                                                ; preds = %2
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !6
  %9 = tail call i64 @strtol(ptr noundef nonnull captures(none) %8, ptr noundef null, i32 noundef 10) #3
  %10 = trunc i64 %9 to i32
  %11 = add nsw i32 %10, -1
  %12 = load i64, ptr @gen_random.last, align 8
  %13 = icmp eq i32 %11, 0
  br i1 %13, label %25, label %14

14:                                               ; preds = %4, %6
  %15 = phi i32 [ %11, %6 ], [ 399999999, %4 ]
  %16 = phi i64 [ %12, %6 ], [ %5, %4 ]
  br label %17

17:                                               ; preds = %14, %17
  %18 = phi i32 [ %20, %17 ], [ %15, %14 ]
  %19 = phi i64 [ %23, %17 ], [ %16, %14 ]
  %20 = add nsw i32 %18, -1
  %21 = mul nuw nsw i64 %19, 3877
  %22 = add nuw nsw i64 %21, 29573
  %23 = urem i64 %22, 139968
  %24 = icmp eq i32 %20, 0
  br i1 %24, label %25, label %17, !llvm.loop !11

25:                                               ; preds = %17, %6
  %26 = phi i64 [ %12, %6 ], [ %23, %17 ]
  %27 = mul nuw nsw i64 %26, 3877
  %28 = add nuw nsw i64 %27, 29573
  %29 = urem i64 %28, 139968
  store i64 %29, ptr @gen_random.last, align 8, !tbaa !13
  %30 = mul nuw nsw i64 %29, 100
  %31 = uitofp nneg i64 %30 to double
  %32 = fdiv double %31, 1.399680e+05
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %32)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!14, !14, i64 0}
!14 = !{!"long", !9, i64 0}
