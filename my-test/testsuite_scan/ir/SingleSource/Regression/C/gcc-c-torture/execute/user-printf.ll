; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/user-printf.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/user-printf.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@stdout = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [2 x i8] c"w\00", align 1
@.str.1 = private unnamed_addr constant [18 x i8] c"fopen for writing\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"1\00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c"%c\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"%c%c\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@.str.6 = private unnamed_addr constant [2 x i8] c"5\00", align 1
@.str.7 = private unnamed_addr constant [5 x i8] c"%s%s\00", align 1
@.str.8 = private unnamed_addr constant [2 x i8] c"6\00", align 1
@.str.9 = private unnamed_addr constant [2 x i8] c"7\00", align 1
@.str.10 = private unnamed_addr constant [3 x i8] c"%i\00", align 1
@.str.11 = private unnamed_addr constant [6 x i8] c"%.1s\0A\00", align 1
@.str.12 = private unnamed_addr constant [3 x i8] c"9x\00", align 1
@.str.13 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str.14 = private unnamed_addr constant [18 x i8] c"fopen for reading\00", align 1
@.str.15 = private unnamed_addr constant [7 x i8] c"fscanf\00", align 1
@.str.16 = private unnamed_addr constant [10 x i8] c"123456789\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @user_print(ptr noundef readonly captures(none) %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  %3 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #12
  call void @llvm.va_start.p0(ptr nonnull %2)
  %4 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %5 = call i32 @vfprintf(ptr noundef %4, ptr noundef %0, ptr dead_on_return noundef nonnull %3) #12
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #12
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #12
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: nofree nounwind
declare noundef i32 @vfprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ptr dead_on_return noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #2

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #5 {
  %1 = alloca [12 x i8], align 1
  %2 = tail call ptr @tmpnam(ptr noundef null) #12
  %3 = load ptr, ptr @stdout, align 8, !tbaa !6
  %4 = tail call ptr @freopen(ptr noundef %2, ptr noundef nonnull @.str, ptr noundef %3) #12
  %5 = icmp eq ptr %4, null
  br i1 %5, label %6, label %7

6:                                                ; preds = %0
  tail call void @perror(ptr noundef nonnull @.str.1) #13
  br label %27

7:                                                ; preds = %0
  tail call void (ptr, ...) @user_print(ptr noundef nonnull @.str.2)
  tail call void (ptr, ...) @user_print(ptr noundef nonnull @.str.3, i32 noundef 50)
  tail call void (ptr, ...) @user_print(ptr noundef nonnull @.str.4, i32 noundef 51, i32 noundef 52)
  tail call void (ptr, ...) @user_print(ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.6)
  tail call void (ptr, ...) @user_print(ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.9)
  tail call void (ptr, ...) @user_print(ptr noundef nonnull @.str.10, i32 noundef 8)
  tail call void (ptr, ...) @user_print(ptr noundef nonnull @.str.11, ptr noundef nonnull @.str.12)
  %8 = tail call i32 @fclose(ptr noundef nonnull %4)
  %9 = tail call noalias ptr @fopen(ptr noundef %2, ptr noundef nonnull @.str.13)
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %13

11:                                               ; preds = %7
  tail call void @perror(ptr noundef nonnull @.str.14) #13
  %12 = tail call i32 @remove(ptr noundef %2) #12
  br label %27

13:                                               ; preds = %7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, i8 0, i64 12, i1 false)
  %14 = call i32 (ptr, ptr, ...) @__isoc99_fscanf(ptr noundef nonnull %9, ptr noundef nonnull @.str.5, ptr noundef nonnull %1) #12
  %15 = icmp eq i32 %14, 1
  br i1 %15, label %19, label %16

16:                                               ; preds = %13
  call void @perror(ptr noundef nonnull @.str.15) #13
  %17 = call i32 @fclose(ptr noundef nonnull %9)
  %18 = call i32 @remove(ptr noundef %2) #12
  br label %25

19:                                               ; preds = %13
  %20 = call i32 @fclose(ptr noundef nonnull %9)
  %21 = call i32 @remove(ptr noundef %2) #12
  %22 = call i32 @bcmp(ptr noundef nonnull dereferenceable(10) %1, ptr noundef nonnull dereferenceable(10) @.str.16, i64 10)
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %25, label %24

24:                                               ; preds = %19
  call void @abort() #14
  unreachable

25:                                               ; preds = %19, %16
  %26 = phi i32 [ 1, %16 ], [ 0, %19 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #12
  br label %27

27:                                               ; preds = %25, %11, %6
  %28 = phi i32 [ %26, %25 ], [ 1, %11 ], [ 1, %6 ]
  ret i32 %28
}

; Function Attrs: nounwind
declare ptr @tmpnam(ptr noundef) local_unnamed_addr #6

declare ptr @freopen(ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #7

; Function Attrs: cold nofree nounwind
declare void @perror(ptr noundef readonly captures(none)) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @fclose(ptr noundef captures(none)) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noalias noundef ptr @fopen(ptr noundef readonly captures(none), ptr noundef readonly captures(none)) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @remove(ptr noundef readonly captures(none)) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #9

declare i32 @__isoc99_fscanf(ptr noundef, ptr noundef, ...) local_unnamed_addr #7

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #10

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #11

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { cold nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #10 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #12 = { nounwind }
attributes #13 = { cold }
attributes #14 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS8_IO_FILE", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i64 0, i64 8, !12, i64 8, i64 8, !12, i64 16, i64 8, !12, i64 24, i64 4, !13, i64 28, i64 4, !13}
!12 = !{!8, !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !9, i64 0}
