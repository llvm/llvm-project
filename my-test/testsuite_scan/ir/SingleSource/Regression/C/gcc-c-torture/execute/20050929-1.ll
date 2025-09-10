; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050929-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050929-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A = type { i32, i32 }
%struct.B = type { ptr, ptr }
%struct.C = type { ptr, ptr }

@.compoundliteral = internal global %struct.A { i32 1, i32 2 }, align 4
@.compoundliteral.1 = internal global %struct.A { i32 3, i32 4 }, align 4
@.compoundliteral.2 = internal global %struct.B { ptr @.compoundliteral, ptr @.compoundliteral.1 }, align 8
@.compoundliteral.3 = internal global %struct.A { i32 5, i32 6 }, align 4
@e = dso_local local_unnamed_addr global %struct.C { ptr @.compoundliteral.2, ptr @.compoundliteral.3 }, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load ptr, ptr @e, align 8, !tbaa !6
  %2 = load ptr, ptr %1, align 8, !tbaa !13
  %3 = load i32, ptr %2, align 4, !tbaa !15
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %5, label %9

5:                                                ; preds = %0
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %7 = load i32, ptr %6, align 4, !tbaa !18
  %8 = icmp eq i32 %7, 2
  br i1 %8, label %10, label %9

9:                                                ; preds = %5, %0
  tail call void @abort() #2
  unreachable

10:                                               ; preds = %5
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load ptr, ptr %11, align 8, !tbaa !19
  %13 = load i32, ptr %12, align 4, !tbaa !15
  %14 = icmp eq i32 %13, 3
  br i1 %14, label %15, label %19

15:                                               ; preds = %10
  %16 = getelementptr inbounds nuw i8, ptr %12, i64 4
  %17 = load i32, ptr %16, align 4, !tbaa !18
  %18 = icmp eq i32 %17, 4
  br i1 %18, label %20, label %19

19:                                               ; preds = %15, %10
  tail call void @abort() #2
  unreachable

20:                                               ; preds = %15
  %21 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @e, i64 8), align 8, !tbaa !20
  %22 = load i32, ptr %21, align 4, !tbaa !15
  %23 = icmp eq i32 %22, 5
  br i1 %23, label %24, label %28

24:                                               ; preds = %20
  %25 = getelementptr inbounds nuw i8, ptr %21, i64 4
  %26 = load i32, ptr %25, align 4, !tbaa !18
  %27 = icmp eq i32 %26, 6
  br i1 %27, label %29, label %28

28:                                               ; preds = %24, %20
  tail call void @abort() #2
  unreachable

29:                                               ; preds = %24
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
!6 = !{!7, !8, i64 0}
!7 = !{!"C", !8, i64 0, !12, i64 8}
!8 = !{!"p1 _ZTS1B", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!"p1 _ZTS1A", !9, i64 0}
!13 = !{!14, !12, i64 0}
!14 = !{!"B", !12, i64 0, !12, i64 8}
!15 = !{!16, !17, i64 0}
!16 = !{!"A", !17, i64 0, !17, i64 4}
!17 = !{!"int", !10, i64 0}
!18 = !{!16, !17, i64 4}
!19 = !{!14, !12, i64 8}
!20 = !{!7, !12, i64 8}
