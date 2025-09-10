; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/aarch64-init-cpu-features.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/aarch64-init-cpu-features.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.anon = type { i64 }

@__aarch64_cpu_features = external local_unnamed_addr global %struct.anon, align 8
@.str = private unnamed_addr constant [43 x i8] c"FAILED consistency test: 0x%llx != 0x%llx\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  tail call void @RUNTIME_INIT() #3
  %1 = load i64, ptr @__aarch64_cpu_features, align 8, !tbaa !6
  store i64 0, ptr @__aarch64_cpu_features, align 8, !tbaa !6
  tail call void @RUNTIME_INIT() #3
  %2 = load i64, ptr @__aarch64_cpu_features, align 8, !tbaa !6
  %3 = icmp eq i64 %2, %1
  br i1 %3, label %6, label %4

4:                                                ; preds = %0
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %1, i64 noundef %2)
  br label %10

6:                                                ; preds = %0, %6
  %7 = phi i32 [ %8, %6 ], [ 0, %0 ]
  store i64 0, ptr @__aarch64_cpu_features, align 8, !tbaa !6
  tail call void @RUNTIME_INIT() #3
  %8 = add nuw nsw i32 %7, 1
  %9 = icmp eq i32 %8, 1000000
  br i1 %9, label %10, label %6, !llvm.loop !11

10:                                               ; preds = %6, %4
  %11 = phi i32 [ 1, %4 ], [ 0, %6 ]
  ret i32 %11
}

declare void @RUNTIME_INIT() local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"", !8, i64 0}
!8 = !{!"long long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
