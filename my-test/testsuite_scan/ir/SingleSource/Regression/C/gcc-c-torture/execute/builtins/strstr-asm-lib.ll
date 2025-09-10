; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strstr-asm-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strstr-asm-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4
@p = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [12 x i8] c"hello world\00", align 1
@llvm.compiler.used = appending global [1 x ptr] [ptr @my_strstr], section "llvm.metadata"

; Function Attrs: nofree nounwind uwtable
define dso_local noundef ptr @my_strstr(ptr noundef readonly %0, ptr noundef readonly captures(address) %1) #0 {
  %3 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1) #3
  %4 = load i32, ptr @inside_main, align 4, !tbaa !6
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = load ptr, ptr @p, align 8, !tbaa !10
  %8 = icmp eq ptr %1, %7
  br i1 %8, label %9, label %12

9:                                                ; preds = %6
  %10 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %0, ptr noundef nonnull dereferenceable(10) getelementptr inbounds nuw (i8, ptr @.str, i64 2)) #3
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %13, label %12

12:                                               ; preds = %9, %6
  tail call void @abort() #4
  unreachable

13:                                               ; preds = %9, %2
  %14 = icmp eq i64 %3, 0
  br i1 %14, label %28, label %15

15:                                               ; preds = %13
  %16 = load i8, ptr %1, align 1, !tbaa !13
  %17 = zext i8 %16 to i32
  %18 = tail call ptr @strchr(ptr noundef nonnull dereferenceable(1) %0, i32 noundef %17) #3
  %19 = icmp eq ptr %18, null
  br i1 %19, label %28, label %20

20:                                               ; preds = %15, %24
  %21 = phi ptr [ %26, %24 ], [ %18, %15 ]
  %22 = tail call i32 @strncmp(ptr noundef nonnull %21, ptr noundef nonnull %1, i64 noundef %3) #3
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %28, label %24

24:                                               ; preds = %20
  %25 = getelementptr inbounds nuw i8, ptr %21, i64 1
  %26 = tail call ptr @strchr(ptr noundef nonnull dereferenceable(1) %25, i32 noundef %17) #3
  %27 = icmp eq ptr %26, null
  br i1 %27, label %28, label %20, !llvm.loop !14

28:                                               ; preds = %20, %24, %15, %13
  %29 = phi ptr [ %0, %13 ], [ null, %15 ], [ %21, %20 ], [ null, %24 ]
  ret ptr %29
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare ptr @strchr(ptr noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strncmp(ptr noundef captures(none), ptr noundef captures(none), i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef ptr @strstr(ptr noundef %0, ptr noundef readonly captures(address) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @inside_main, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  tail call void @abort() #4
  unreachable

6:                                                ; preds = %2
  %7 = tail call ptr @my_strstr(ptr noundef %0, ptr noundef %1)
  ret ptr %7
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
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
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 omnipotent char", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!8, !8, i64 0}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
