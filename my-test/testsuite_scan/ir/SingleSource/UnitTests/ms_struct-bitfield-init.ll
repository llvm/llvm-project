; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/ms_struct-bitfield-init.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/ms_struct-bitfield-init.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.anon = type { i8, i8 }
%struct.anon.0 = type { i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.anon.1 = type { i32, i32, i32, i32 }

@t1 = dso_local local_unnamed_addr global %struct.anon { i8 97, i8 98 }, align 4
@t2 = dso_local local_unnamed_addr global %struct.anon.0 { i8 97, i8 98, i8 99, i8 100, i8 101, i8 102, i8 103, i8 104, i8 105 }, align 4
@t3 = dso_local local_unnamed_addr global %struct.anon.1 { i32 1, i32 2, i32 3, i32 4 }, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i8, ptr @t1, align 4, !tbaa !6
  %2 = icmp eq i8 %1, 97
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #2
  unreachable

4:                                                ; preds = %0
  %5 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t1, i64 1), align 1, !tbaa !10
  %6 = icmp eq i8 %5, 98
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #2
  unreachable

8:                                                ; preds = %4
  store i8 99, ptr @t1, align 4, !tbaa !6
  store i8 100, ptr getelementptr inbounds nuw (i8, ptr @t1, i64 1), align 1, !tbaa !10
  %9 = load i8, ptr @t2, align 4, !tbaa !11
  %10 = icmp ne i8 %9, 97
  %11 = load i8, ptr getelementptr inbounds nuw (i8, ptr @t2, i64 8), align 4
  %12 = icmp ne i8 %11, 105
  %13 = select i1 %10, i1 true, i1 %12
  br i1 %13, label %14, label %15

14:                                               ; preds = %8
  tail call void @abort() #2
  unreachable

15:                                               ; preds = %8
  %16 = load i32, ptr @t3, align 4, !tbaa !13
  %17 = icmp ne i32 %16, 1
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @t3, i64 12), align 4
  %19 = icmp ne i32 %18, 4
  %20 = select i1 %17, i1 true, i1 %19
  br i1 %20, label %21, label %22

21:                                               ; preds = %15
  tail call void @abort() #2
  unreachable

22:                                               ; preds = %15
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"", !8, i64 0, !8, i64 1}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!7, !8, i64 1}
!11 = !{!12, !8, i64 0}
!12 = !{!"", !8, i64 0, !8, i64 1, !8, i64 2, !8, i64 3, !8, i64 4, !8, i64 5, !8, i64 6, !8, i64 7, !8, i64 8}
!13 = !{!14, !15, i64 0}
!14 = !{!"", !15, i64 0, !15, i64 4, !15, i64 8, !15, i64 12}
!15 = !{!"int", !8, i64 0}
