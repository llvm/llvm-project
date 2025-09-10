; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20051110-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20051110-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@bytes = dso_local local_unnamed_addr global [5 x i8] zeroinitializer, align 4
@flag = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @add_unwind_adjustsp(i64 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i64 %0, -516
  %3 = ashr i64 %2, 2
  %4 = load i32, ptr @flag, align 4
  %5 = freeze i32 %4
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %18

7:                                                ; preds = %1
  %8 = icmp ult i64 %3, 128
  br i1 %8, label %24, label %9

9:                                                ; preds = %7, %9
  %10 = phi i64 [ %16, %9 ], [ 0, %7 ]
  %11 = phi i64 [ %13, %9 ], [ %3, %7 ]
  %12 = getelementptr inbounds nuw i8, ptr @bytes, i64 %10
  %13 = lshr i64 %11, 7
  %14 = trunc i64 %11 to i8
  %15 = or i8 %14, -128
  store i8 %15, ptr %12, align 1, !tbaa !6
  %16 = add nuw nsw i64 %10, 1
  %17 = icmp ult i64 %11, 16384
  br i1 %17, label %22, label %9

18:                                               ; preds = %1, %18
  %19 = phi i64 [ %20, %18 ], [ %3, %1 ]
  %20 = lshr i64 %19, 7
  %21 = icmp ult i64 %19, 128
  br i1 %21, label %24, label %18

22:                                               ; preds = %9
  %23 = getelementptr inbounds nuw i8, ptr @bytes, i64 %16
  br label %24

24:                                               ; preds = %18, %22, %7
  %25 = phi i64 [ %3, %7 ], [ %13, %22 ], [ %19, %18 ]
  %26 = phi ptr [ @bytes, %7 ], [ %23, %22 ], [ @bytes, %18 ]
  %27 = trunc i64 %25 to i8
  %28 = and i8 %27, 127
  store i8 %28, ptr %26, align 1, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @flag, align 4
  %2 = freeze i32 %1
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  store i8 -120, ptr @bytes, align 4, !tbaa !6
  br label %5

5:                                                ; preds = %0, %4
  %6 = phi ptr [ getelementptr inbounds nuw (i8, ptr @bytes, i64 1), %4 ], [ @bytes, %0 ]
  store i8 7, ptr %6, align 1, !tbaa !6
  %7 = load i8, ptr @bytes, align 4, !tbaa !6
  %8 = icmp ne i8 %7, -120
  %9 = load i8, ptr getelementptr inbounds nuw (i8, ptr @bytes, i64 1), align 1
  %10 = icmp ne i8 %9, 7
  %11 = select i1 %8, i1 true, i1 %10
  br i1 %11, label %12, label %13

12:                                               ; preds = %5
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %5
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
