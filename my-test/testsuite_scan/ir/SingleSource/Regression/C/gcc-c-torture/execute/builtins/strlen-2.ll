; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strlen-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strlen-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@l = dso_local local_unnamed_addr global i64 0, align 8
@i = dso_local local_unnamed_addr global i64 0, align 8
@.str = private unnamed_addr constant [4 x i8] c"foo\00", align 1
@j = dso_local local_unnamed_addr global i64 0, align 8
@.str.1 = private unnamed_addr constant [4 x i8] c"bar\00", align 1
@g = dso_local local_unnamed_addr global i64 0, align 8
@h = dso_local local_unnamed_addr global i64 0, align 8
@inside_main = external local_unnamed_addr global i32, align 4
@k = dso_local local_unnamed_addr global i64 0, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i64 @foo() local_unnamed_addr #0 {
  %1 = load i64, ptr @l, align 8, !tbaa !6
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %0
  store i64 1, ptr @l, align 8, !tbaa !6
  ret i64 1
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = load i64, ptr @i, align 8, !tbaa !6
  %2 = load i64, ptr @g, align 8, !tbaa !6
  %3 = add i64 %2, 1
  store i64 %3, ptr @g, align 8, !tbaa !6
  %4 = icmp eq i64 %2, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

6:                                                ; preds = %0
  %7 = load i64, ptr @h, align 8, !tbaa !6
  %8 = add i64 %7, 1
  store i64 %8, ptr @h, align 8, !tbaa !6
  %9 = icmp eq i64 %7, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @abort() #3
  unreachable

11:                                               ; preds = %6
  %12 = add i64 %1, 1
  store i64 %12, ptr @i, align 8, !tbaa !6
  %13 = icmp eq i64 %1, 0
  br i1 %13, label %15, label %14

14:                                               ; preds = %11
  tail call void @abort() #3
  unreachable

15:                                               ; preds = %11
  store i32 0, ptr @inside_main, align 4, !tbaa !10
  %16 = load i64, ptr @j, align 8, !tbaa !6
  %17 = icmp eq i64 %16, 0
  %18 = load i64, ptr @k, align 8, !tbaa !6
  %19 = select i1 %17, ptr @.str.1, ptr @.str
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 %18
  %21 = add i64 %18, 1
  store i64 %21, ptr @k, align 8, !tbaa !6
  %22 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %20) #4
  %23 = icmp ne i64 %22, 3
  %24 = icmp ne i64 %18, 0
  %25 = or i1 %23, %24
  br i1 %25, label %26, label %27

26:                                               ; preds = %15
  tail call void @abort() #3
  unreachable

27:                                               ; preds = %15
  %28 = load i64, ptr @l, align 8, !tbaa !6
  %29 = icmp eq i64 %28, 0
  br i1 %29, label %31, label %30

30:                                               ; preds = %27
  tail call void @abort() #3
  unreachable

31:                                               ; preds = %27
  store i64 1, ptr @l, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
