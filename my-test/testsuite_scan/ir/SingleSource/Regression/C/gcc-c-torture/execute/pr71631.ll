; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr71631.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr71631.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 1, align 4
@b = dso_local local_unnamed_addr global i32 1, align 4
@c = dso_local local_unnamed_addr global i32 1, align 4
@v = dso_local global i8 0, align 4

; Function Attrs: nofree norecurse nounwind memory(readwrite, argmem: read) uwtable
define dso_local void @foo(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i8, ptr %0, align 1, !tbaa !6
  %3 = icmp eq i8 %2, 0
  br i1 %3, label %9, label %4

4:                                                ; preds = %1, %4
  %5 = phi ptr [ %6, %4 ], [ %0, %1 ]
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 1
  %7 = load i8, ptr %6, align 1, !tbaa !6
  store volatile i8 %7, ptr @v, align 4, !tbaa !6
  %8 = icmp eq i8 %7, 0
  br i1 %8, label %9, label %4, !llvm.loop !9

9:                                                ; preds = %4, %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store volatile i32 1, ptr %1, align 4, !tbaa !11
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  store volatile i32 1, ptr %2, align 4, !tbaa !11
  %3 = load i32, ptr @a, align 4, !tbaa !11
  %4 = freeze i32 %3
  %5 = icmp eq i32 %4, -1
  %6 = load volatile i32, ptr %2, align 4, !tbaa !11
  %7 = load i32, ptr @b, align 4
  %8 = load i32, ptr @c, align 4, !tbaa !11
  %9 = load volatile i32, ptr %1, align 4, !tbaa !11
  br i1 %5, label %18, label %10

10:                                               ; preds = %0
  %11 = icmp ne i32 %6, 0
  %12 = icmp ne i32 %7, 0
  %13 = select i1 %11, i1 %12, i1 false
  %14 = icmp eq i32 %8, 1
  %15 = icmp ne i32 %9, 0
  %16 = and i1 %14, %15
  %17 = and i1 %13, %16
  store volatile i8 66, ptr @v, align 4, !tbaa !6
  store volatile i8 0, ptr @v, align 4, !tbaa !6
  br i1 %17, label %19, label %18

18:                                               ; preds = %0, %10
  tail call void @abort() #4
  unreachable

19:                                               ; preds = %10
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

attributes #0 = { nofree norecurse nounwind memory(readwrite, argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !7, i64 0}
