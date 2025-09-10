; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strpcpy-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strpcpy-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@buf1 = dso_local global [64 x i64] zeroinitializer, align 8
@buf2 = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @buf1, i64 256), align 8
@.str = private unnamed_addr constant [17 x i8] c"abcdefghijklmnop\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"ABCDEFG\00ijklmnop\00", align 1
@.str.4 = private unnamed_addr constant [17 x i8] c"ABCDx\00G\00ijklmnop\00", align 1
@inside_main = external local_unnamed_addr global i32, align 4
@buf5 = dso_local local_unnamed_addr global [20 x i64] zeroinitializer, align 8
@.str.5 = private unnamed_addr constant [20 x i8] c"RSTUVWXYZ0123456789\00", align 1
@buf7 = dso_local local_unnamed_addr global [20 x i8] zeroinitializer, align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @test(ptr noundef writeonly captures(address) initializes((0, 17)) %0, ptr readnone captures(none) %1, ptr readnone captures(none) %2, i32 %3) local_unnamed_addr #0 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(17) %0, ptr noundef nonnull align 1 dereferenceable(17) @.str, i64 17, i1 false) #5
  %5 = icmp eq ptr %0, @buf1
  br i1 %5, label %6, label %9

6:                                                ; preds = %4
  %7 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(17) @buf1, ptr noundef nonnull dereferenceable(17) @.str, i64 17)
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %6, %4
  tail call void @abort() #6
  unreachable

10:                                               ; preds = %6
  store i64 20061986658402881, ptr @buf1, align 8
  %11 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(17) @buf1, ptr noundef nonnull dereferenceable(17) @.str.2, i64 17)
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %14, label %13

13:                                               ; preds = %10
  tail call void @abort() #6
  unreachable

14:                                               ; preds = %10
  store i16 120, ptr getelementptr inbounds nuw (i8, ptr @buf1, i64 4), align 4
  %15 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(17) @buf1, ptr noundef nonnull dereferenceable(17) @.str.4, i64 17)
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14
  tail call void @abort() #6
  unreachable

18:                                               ; preds = %14
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #2 {
  store i32 0, ptr @inside_main, align 4, !tbaa !6
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(20) @buf5, ptr noundef nonnull align 1 dereferenceable(20) @.str.5, i64 20, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(20) @buf7, ptr noundef nonnull align 1 dereferenceable(20) @.str.5, i64 20, i1 false)
  tail call void @test(ptr noundef nonnull @buf1, ptr poison, ptr nonnull poison, i32 poison)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #4

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

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
