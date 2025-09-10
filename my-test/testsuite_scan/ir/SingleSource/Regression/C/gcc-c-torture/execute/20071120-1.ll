; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071120-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071120-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.ggc_root_tab = type { ptr }

@deferred_access_no_check = internal global i32 0, align 4
@gt_pch_rs_gt_cp_semantics_h = dso_local local_unnamed_addr constant [1 x %struct.ggc_root_tab] [%struct.ggc_root_tab { ptr @deferred_access_no_check }], align 8
@deferred_access_stack.init = internal unnamed_addr global i1 false
@deferred_access_stack.body = internal unnamed_addr global [152 x i8] undef

; Function Attrs: cold nofree noinline noreturn nounwind uwtable
define dso_local void @vec_assert_fail() local_unnamed_addr #0 {
  tail call void @abort() #4
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: cold nofree noinline noreturn nounwind uwtable
define dso_local void @perform_access_checks(ptr readnone captures(none) %0) local_unnamed_addr #0 {
  tail call void @abort() #4
  unreachable
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @pop_to_parent_deferring_access_checks() local_unnamed_addr #2 {
  %1 = load i32, ptr @deferred_access_no_check, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %5, label %3

3:                                                ; preds = %0
  %4 = add i32 %1, -1
  store i32 %4, ptr @deferred_access_no_check, align 4, !tbaa !6
  br label %24

5:                                                ; preds = %0
  %6 = load i1, ptr @deferred_access_stack.init, align 1
  br i1 %6, label %7, label %10

7:                                                ; preds = %5
  %8 = load i32, ptr @deferred_access_stack.body, align 16, !tbaa !10
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %10, label %11

10:                                               ; preds = %7, %5
  tail call void @vec_assert_fail() #5
  unreachable

11:                                               ; preds = %7
  %12 = add i32 %8, -1
  store i32 %12, ptr @deferred_access_stack.body, align 16, !tbaa !10
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %15

14:                                               ; preds = %11
  tail call void @vec_assert_fail() #5
  unreachable

15:                                               ; preds = %11
  %16 = add i32 %8, -2
  %17 = zext i32 %16 to i64
  %18 = shl nuw nsw i64 %17, 4
  %19 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @deferred_access_stack.body, i64 8), i64 %18
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 8
  %21 = load i32, ptr %20, align 16, !tbaa !12
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %23, label %24

23:                                               ; preds = %15
  tail call void @perform_access_checks(ptr poison)
  unreachable

24:                                               ; preds = %15, %3
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  store i1 true, ptr @deferred_access_stack.init, align 1
  store i32 2, ptr @deferred_access_stack.body, align 16, !tbaa !16
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @deferred_access_stack.body, i64 16), align 16, !tbaa !12
  tail call void @pop_to_parent_deferring_access_checks()
  ret i32 0
}

attributes #0 = { cold nofree noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }
attributes #5 = { noreturn }

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
!10 = !{!11, !7, i64 0}
!11 = !{!"VEC_deferred_access_base", !7, i64 0, !8, i64 8}
!12 = !{!13, !7, i64 8}
!13 = !{!"deferred_access", !14, i64 0, !7, i64 8}
!14 = !{!"p1 _ZTS21deferred_access_check", !15, i64 0}
!15 = !{!"any pointer", !8, i64 0}
!16 = !{!17, !7, i64 0}
!17 = !{!"VEC_deferred_access_gc", !11, i64 0}
