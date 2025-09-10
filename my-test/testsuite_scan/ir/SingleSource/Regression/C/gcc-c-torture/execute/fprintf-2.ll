; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/fprintf-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/fprintf-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [2 x i8] c"w\00", align 1
@.str.1 = private unnamed_addr constant [18 x i8] c"fopen for writing\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"%c%c\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%s\00", align 1
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

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  %1 = alloca [12 x i8], align 1
  %2 = tail call ptr @tmpnam(ptr noundef null) #10
  %3 = tail call noalias ptr @fopen(ptr noundef %2, ptr noundef nonnull @.str)
  %4 = icmp eq ptr %3, null
  br i1 %4, label %5, label %6

5:                                                ; preds = %0
  tail call void @perror(ptr noundef nonnull @.str.1) #11
  br label %33

6:                                                ; preds = %0
  %7 = tail call i32 @fputc(i32 49, ptr nonnull %3)
  %8 = tail call i32 @fputc(i32 50, ptr nonnull %3)
  %9 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef nonnull %3, ptr noundef nonnull @.str.4, i32 noundef 51, i32 noundef 52) #10
  %10 = tail call i32 @fputc(i32 53, ptr nonnull %3)
  %11 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef nonnull %3, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.9) #10
  %12 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef nonnull %3, ptr noundef nonnull @.str.10, i32 noundef 8) #10
  %13 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef nonnull %3, ptr noundef nonnull @.str.11, ptr noundef nonnull @.str.12) #10
  %14 = tail call i32 @fclose(ptr noundef nonnull %3)
  %15 = tail call noalias ptr @fopen(ptr noundef %2, ptr noundef nonnull @.str.13)
  %16 = icmp eq ptr %15, null
  br i1 %16, label %17, label %19

17:                                               ; preds = %6
  tail call void @perror(ptr noundef nonnull @.str.14) #11
  %18 = tail call i32 @remove(ptr noundef %2) #10
  br label %33

19:                                               ; preds = %6
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #10
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, i8 0, i64 12, i1 false)
  %20 = call i32 (ptr, ptr, ...) @__isoc99_fscanf(ptr noundef nonnull %15, ptr noundef nonnull @.str.5, ptr noundef nonnull %1) #10
  %21 = icmp eq i32 %20, 1
  br i1 %21, label %25, label %22

22:                                               ; preds = %19
  call void @perror(ptr noundef nonnull @.str.15) #11
  %23 = call i32 @fclose(ptr noundef nonnull %15)
  %24 = call i32 @remove(ptr noundef %2) #10
  br label %31

25:                                               ; preds = %19
  %26 = call i32 @fclose(ptr noundef nonnull %15)
  %27 = call i32 @remove(ptr noundef %2) #10
  %28 = call i32 @bcmp(ptr noundef nonnull dereferenceable(10) %1, ptr noundef nonnull dereferenceable(10) @.str.16, i64 10)
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %31, label %30

30:                                               ; preds = %25
  call void @abort() #12
  unreachable

31:                                               ; preds = %25, %22
  %32 = phi i32 [ 1, %22 ], [ 0, %25 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #10
  br label %33

33:                                               ; preds = %31, %17, %5
  %34 = phi i32 [ %32, %31 ], [ 1, %17 ], [ 1, %5 ]
  ret i32 %34
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nounwind
declare ptr @tmpnam(ptr noundef) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noalias noundef ptr @fopen(ptr noundef readonly captures(none), ptr noundef readonly captures(none)) local_unnamed_addr #3

; Function Attrs: cold nofree nounwind
declare void @perror(ptr noundef readonly captures(none)) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @fclose(ptr noundef captures(none)) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @remove(ptr noundef readonly captures(none)) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

declare i32 @__isoc99_fscanf(ptr noundef, ptr noundef, ...) local_unnamed_addr #6

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind
declare noundef i32 @fputc(i32 noundef, ptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #9

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree nounwind }
attributes #9 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #10 = { nounwind }
attributes #11 = { cold }
attributes #12 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
