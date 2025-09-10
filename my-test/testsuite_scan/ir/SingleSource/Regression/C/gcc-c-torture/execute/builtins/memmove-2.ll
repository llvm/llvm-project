; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memmove-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memmove-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@p = dso_local global [32 x i8] c"abcdefg\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1
@q = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @p, i64 4), align 8
@.str = private unnamed_addr constant [8 x i8] c"abddefg\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"abddeff\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 3), align 1
  store i8 %1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 2), align 1
  %2 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(8) @p, ptr noundef nonnull dereferenceable(8) @.str, i64 8)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %0
  %6 = load ptr, ptr @q, align 8, !tbaa !6
  %7 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4), align 1
  store i8 %7, ptr %6, align 1
  %8 = icmp eq ptr %6, getelementptr inbounds nuw (i8, ptr @p, i64 4)
  br i1 %8, label %9, label %12

9:                                                ; preds = %5
  %10 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(8) @p, ptr noundef nonnull dereferenceable(8) @.str, i64 8)
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %13, label %12

12:                                               ; preds = %9, %5
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %9
  %14 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 5), align 1
  store i8 %14, ptr getelementptr inbounds nuw (i8, ptr @p, i64 6), align 1
  %15 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(8) @p, ptr noundef nonnull dereferenceable(8) @.str.1, i64 8)
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %13
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %13
  %19 = load ptr, ptr @q, align 8, !tbaa !6
  %20 = load i8, ptr %19, align 1
  store i8 %20, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4), align 1
  %21 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(8) @p, ptr noundef nonnull dereferenceable(8) @.str.1, i64 8)
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %24, label %23

23:                                               ; preds = %18
  tail call void @abort() #3
  unreachable

24:                                               ; preds = %18
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: read) }
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
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
