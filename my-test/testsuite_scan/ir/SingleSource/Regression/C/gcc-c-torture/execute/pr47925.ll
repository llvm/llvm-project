; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr47925.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr47925.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.s = type { ptr }

; Function Attrs: noinline nounwind uwtable
define dso_local void @bar(i32 %0, i32 %1) local_unnamed_addr #0 {
  tail call void asm sideeffect "", ""() #3, !srcloc !6
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @foo(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  tail call void @bar(i32 poison, i32 poison)
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %10

4:                                                ; preds = %2, %4
  %5 = phi i32 [ %8, %4 ], [ 0, %2 ]
  %6 = phi ptr [ %7, %4 ], [ %0, %2 ]
  %7 = load volatile ptr, ptr %6, align 8, !tbaa !7
  %8 = add nuw nsw i32 %5, 1
  %9 = icmp eq i32 %8, %1
  br i1 %9, label %10, label %4, !llvm.loop !13

10:                                               ; preds = %4, %2
  ret i32 undef
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca %struct.s, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #3
  store ptr %1, ptr %1, align 8, !tbaa !7
  %2 = call i32 @foo(ptr noundef nonnull %1, i32 noundef 10)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #3
  ret i32 0
}

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 116}
!7 = !{!8, !9, i64 0}
!8 = !{!"s", !9, i64 0}
!9 = !{!"p1 _ZTS1s", !10, i64 0}
!10 = !{!"any pointer", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
