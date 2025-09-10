; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memmove.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memmove.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@s1 = dso_local local_unnamed_addr constant [4 x i8] c"123\00", align 1
@p = dso_local global [32 x i8] zeroinitializer, align 1
@.str = private unnamed_addr constant [6 x i8] c"abcde\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"abc\00e\00", align 1
@.str.4 = private unnamed_addr constant [7 x i8] c"abfghi\00", align 1
@.str.6 = private unnamed_addr constant [7 x i8] c"abfgAi\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) @p, ptr noundef nonnull align 1 dereferenceable(6) @.str, i64 6, i1 false)
  %1 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str, i64 6)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #4
  unreachable

4:                                                ; preds = %0
  store i8 0, ptr getelementptr inbounds nuw (i8, ptr @p, i64 3), align 1
  %5 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.2, i64 6)
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #4
  unreachable

8:                                                ; preds = %4
  store i32 1768449894, ptr getelementptr inbounds nuw (i8, ptr @p, i64 2), align 1
  %9 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(7) @p, ptr noundef nonnull dereferenceable(7) @.str.4, i64 7)
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #4
  unreachable

12:                                               ; preds = %8
  store i8 65, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4), align 1
  %13 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(7) @p, ptr noundef nonnull dereferenceable(7) @.str.6, i64 7)
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %16, label %15

15:                                               ; preds = %12
  tail call void @abort() #4
  unreachable

16:                                               ; preds = %12
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
