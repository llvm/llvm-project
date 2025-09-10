; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strrchr.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strrchr.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"hi world\00", align 1
@bar = dso_local local_unnamed_addr global ptr @.str, align 8
@x = dso_local local_unnamed_addr global i32 7, align 4
@.str.1 = private unnamed_addr constant [12 x i8] c"hello world\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"hello\00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = load ptr, ptr @bar, align 8, !tbaa !6
  %2 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %3 = icmp eq i64 %2, 8
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #4
  unreachable

5:                                                ; preds = %0
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %7 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %6)
  %8 = icmp eq i64 %7, 4
  br i1 %8, label %10, label %9

9:                                                ; preds = %5
  tail call void @abort() #4
  unreachable

10:                                               ; preds = %5
  %11 = load i32, ptr @x, align 4, !tbaa !11
  %12 = add nsw i32 %11, 1
  store i32 %12, ptr @x, align 4, !tbaa !11
  %13 = and i32 %11, 3
  %14 = zext nneg i32 %13 to i64
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 %14
  %16 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %15)
  %17 = add nsw i64 %16, %14
  %18 = icmp eq i64 %17, 8
  br i1 %18, label %20, label %19

19:                                               ; preds = %10
  tail call void @abort() #4
  unreachable

20:                                               ; preds = %10
  %21 = icmp eq i32 %12, 8
  br i1 %21, label %23, label %22

22:                                               ; preds = %20
  tail call void @abort() #4
  unreachable

23:                                               ; preds = %20
  %24 = tail call ptr @rindex(ptr noundef nonnull @.str.2, i32 noundef 122) #5
  %25 = icmp eq ptr %24, null
  br i1 %25, label %27, label %26

26:                                               ; preds = %23
  tail call void @abort() #4
  unreachable

27:                                               ; preds = %23
  %28 = tail call ptr @rindex(ptr noundef nonnull @.str.1, i32 noundef 111) #5
  %29 = icmp eq ptr %28, getelementptr inbounds nuw (i8, ptr @.str.1, i64 7)
  br i1 %29, label %31, label %30

30:                                               ; preds = %27
  tail call void @abort() #4
  unreachable

31:                                               ; preds = %27
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nounwind
declare ptr @rindex(ptr noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr captures(none)) local_unnamed_addr #3

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { noreturn nounwind }
attributes #5 = { nounwind }

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
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
