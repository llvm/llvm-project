; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/noinit-attribute.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/noinit-attribute.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@var_zero = dso_local local_unnamed_addr global i32 0, align 4
@var_one = dso_local local_unnamed_addr global i32 1, align 4
@var_init = dso_local local_unnamed_addr global i32 2, align 4
@var_common = dso_local local_unnamed_addr global i32 0, align 4
@var_noinit = dso_local local_unnamed_addr global i32 0, align 4
@var_section1 = dso_local local_unnamed_addr global i32 0, section "mysection", align 4
@var_section2 = dso_local local_unnamed_addr global i32 0, section "mysection", align 4

; Function Attrs: noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @var_common, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #4
  unreachable

4:                                                ; preds = %0
  %5 = load i32, ptr @var_init, align 4, !tbaa !6
  switch i32 %5, label %19 [
    i32 2, label %6
    i32 3, label %14
  ]

6:                                                ; preds = %4
  %7 = load i32, ptr @var_zero, align 4, !tbaa !6
  %8 = icmp ne i32 %7, 0
  %9 = load i32, ptr @var_one, align 4
  %10 = icmp ne i32 %9, 1
  %11 = select i1 %8, i1 true, i1 %10
  br i1 %11, label %12, label %13

12:                                               ; preds = %6
  tail call void @abort() #4
  unreachable

13:                                               ; preds = %6
  store i32 3, ptr @var_init, align 4, !tbaa !6
  store i32 3, ptr @var_noinit, align 4, !tbaa !6
  store i32 3, ptr @var_one, align 4, !tbaa !6
  store i32 3, ptr @var_zero, align 4, !tbaa !6
  store i32 3, ptr @var_common, align 4, !tbaa !6
  tail call void @_start() #4
  unreachable

14:                                               ; preds = %4
  %15 = load i32, ptr @var_noinit, align 4, !tbaa !6
  %16 = icmp eq i32 %15, 3
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  tail call void @abort() #4
  unreachable

18:                                               ; preds = %14
  tail call void @exit(i32 noundef 0) #4
  unreachable

19:                                               ; preds = %4
  tail call void @abort() #4
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

; Function Attrs: noreturn
declare void @_start() local_unnamed_addr #3

attributes #0 = { noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
