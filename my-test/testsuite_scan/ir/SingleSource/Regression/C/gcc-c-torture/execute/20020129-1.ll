; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020129-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020129-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.B = type { %struct.C, ptr }
%struct.C = type { %struct.D, i64, i64, i64, i64, i64 }
%struct.D = type { i32, ptr }

@y = dso_local global %struct.B zeroinitializer, align 8
@x = dso_local global %struct.B zeroinitializer, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(ptr noundef %0, ptr noundef captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %4 = load ptr, ptr %3, align 8, !tbaa !6
  %5 = icmp eq ptr %4, null
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %7 = load ptr, ptr %6, align 8, !tbaa !6
  br i1 %5, label %8, label %15

8:                                                ; preds = %2
  store ptr %7, ptr %3, align 8, !tbaa !6
  store ptr null, ptr %6, align 8, !tbaa !6
  %9 = icmp eq ptr %7, null
  br i1 %9, label %18, label %10

10:                                               ; preds = %8, %10
  %11 = phi ptr [ %13, %10 ], [ %7, %8 ]
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store ptr %0, ptr %12, align 8, !tbaa !17
  %13 = load ptr, ptr %11, align 8, !tbaa !20
  %14 = icmp eq ptr %13, null
  br i1 %14, label %18, label %10, !llvm.loop !21

15:                                               ; preds = %2
  %16 = icmp eq ptr %7, null
  br i1 %16, label %18, label %17

17:                                               ; preds = %15
  tail call void @abort() #4
  unreachable

18:                                               ; preds = %10, %8, %15
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %20 = load i64, ptr %19, align 8, !tbaa !23
  %21 = icmp eq i64 %20, -1
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %23 = load i64, ptr %22, align 8, !tbaa !23
  br i1 %21, label %24, label %28

24:                                               ; preds = %18
  store i64 %23, ptr %19, align 8, !tbaa !23
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %26 = load i64, ptr %25, align 8, !tbaa !24
  %27 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store i64 %26, ptr %27, align 8, !tbaa !24
  store <2 x i64> <i64 -1, i64 0>, ptr %22, align 8, !tbaa !25
  br label %31

28:                                               ; preds = %18
  %29 = icmp eq i64 %23, -1
  br i1 %29, label %31, label %30

30:                                               ; preds = %28
  tail call void @abort() #4
  unreachable

31:                                               ; preds = %24, %28
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  store i32 6, ptr @y, align 8, !tbaa !26
  store i64 145, ptr getelementptr inbounds nuw (i8, ptr @y, i64 24), align 8, !tbaa !23
  store i64 2448, ptr getelementptr inbounds nuw (i8, ptr @y, i64 32), align 8, !tbaa !24
  store i64 -1, ptr getelementptr inbounds nuw (i8, ptr @x, i64 24), align 8, !tbaa !23
  tail call void @foo(ptr noundef nonnull @x, ptr noundef nonnull @y)
  tail call void @exit(i32 noundef 0) #4
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !16, i64 56}
!7 = !{!"B", !8, i64 0, !16, i64 56}
!8 = !{!"C", !9, i64 0, !15, i64 16, !15, i64 24, !15, i64 32, !15, i64 40, !15, i64 48}
!9 = !{!"D", !10, i64 0, !13, i64 8}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
!13 = !{!"p1 _ZTS1D", !14, i64 0}
!14 = !{!"any pointer", !11, i64 0}
!15 = !{!"long", !11, i64 0}
!16 = !{!"p1 _ZTS1A", !14, i64 0}
!17 = !{!18, !19, i64 8}
!18 = !{!"A", !16, i64 0, !19, i64 8}
!19 = !{!"p1 _ZTS1C", !14, i64 0}
!20 = !{!18, !16, i64 0}
!21 = distinct !{!21, !22}
!22 = !{!"llvm.loop.mustprogress"}
!23 = !{!7, !15, i64 24}
!24 = !{!7, !15, i64 32}
!25 = !{!15, !15, i64 0}
!26 = !{!7, !10, i64 0}
