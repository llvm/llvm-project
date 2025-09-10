; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/printf-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/printf-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str.2 = private unnamed_addr constant [5 x i8] c"%c%c\00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@.str.5 = private unnamed_addr constant [5 x i8] c"%s%s\00", align 1
@.str.6 = private unnamed_addr constant [2 x i8] c"6\00", align 1
@.str.7 = private unnamed_addr constant [2 x i8] c"7\00", align 1
@.str.8 = private unnamed_addr constant [3 x i8] c"%i\00", align 1
@.str.9 = private unnamed_addr constant [6 x i8] c"%.1s\0A\00", align 1
@.str.10 = private unnamed_addr constant [3 x i8] c"9x\00", align 1
@.str.11 = private unnamed_addr constant [2 x i8] c"w\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.12 = private unnamed_addr constant [18 x i8] c"fopen for writing\00", align 1
@.str.13 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str.14 = private unnamed_addr constant [18 x i8] c"fopen for reading\00", align 1
@.str.15 = private unnamed_addr constant [7 x i8] c"fscanf\00", align 1
@.str.16 = private unnamed_addr constant [10 x i8] c"123456789\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @write_file() local_unnamed_addr #0 {
  %1 = tail call i32 @putchar(i32 49)
  %2 = tail call i32 @putchar(i32 50)
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 51, i32 noundef 52)
  %4 = tail call i32 @putchar(i32 53)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.7)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef 8)
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, ptr noundef nonnull @.str.10)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #2 {
  %1 = alloca [12 x i8], align 1
  %2 = tail call ptr @tmpnam(ptr noundef null) #11
  %3 = load ptr, ptr @stdout, align 8, !tbaa !6
  %4 = tail call ptr @freopen(ptr noundef %2, ptr noundef nonnull @.str.11, ptr noundef %3) #11
  %5 = icmp eq ptr %4, null
  br i1 %5, label %6, label %7

6:                                                ; preds = %0
  tail call void @perror(ptr noundef nonnull @.str.12) #12
  br label %34

7:                                                ; preds = %0
  %8 = tail call i32 @putchar(i32 49)
  %9 = tail call i32 @putchar(i32 50)
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 51, i32 noundef 52)
  %11 = tail call i32 @putchar(i32 53)
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.7)
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef 8)
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, ptr noundef nonnull @.str.10)
  %15 = tail call i32 @fclose(ptr noundef nonnull %4)
  %16 = tail call noalias ptr @fopen(ptr noundef %2, ptr noundef nonnull @.str.13)
  %17 = icmp eq ptr %16, null
  br i1 %17, label %18, label %20

18:                                               ; preds = %7
  tail call void @perror(ptr noundef nonnull @.str.14) #12
  %19 = tail call i32 @remove(ptr noundef %2) #11
  br label %34

20:                                               ; preds = %7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #11
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, i8 0, i64 12, i1 false)
  %21 = call i32 (ptr, ptr, ...) @__isoc99_fscanf(ptr noundef nonnull %16, ptr noundef nonnull @.str.3, ptr noundef nonnull %1) #11
  %22 = icmp eq i32 %21, 1
  br i1 %22, label %26, label %23

23:                                               ; preds = %20
  call void @perror(ptr noundef nonnull @.str.15) #12
  %24 = call i32 @fclose(ptr noundef nonnull %16)
  %25 = call i32 @remove(ptr noundef %2) #11
  br label %32

26:                                               ; preds = %20
  %27 = call i32 @fclose(ptr noundef nonnull %16)
  %28 = call i32 @remove(ptr noundef %2) #11
  %29 = call i32 @bcmp(ptr noundef nonnull dereferenceable(10) %1, ptr noundef nonnull dereferenceable(10) @.str.16, i64 10)
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %32, label %31

31:                                               ; preds = %26
  call void @abort() #13
  unreachable

32:                                               ; preds = %26, %23
  %33 = phi i32 [ 1, %23 ], [ 0, %26 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #11
  br label %34

34:                                               ; preds = %32, %18, %6
  %35 = phi i32 [ %33, %32 ], [ 1, %18 ], [ 1, %6 ]
  ret i32 %35
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: nounwind
declare ptr @tmpnam(ptr noundef) local_unnamed_addr #4

declare ptr @freopen(ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #5

; Function Attrs: cold nofree nounwind
declare void @perror(ptr noundef readonly captures(none)) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare noundef i32 @fclose(ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noalias noundef ptr @fopen(ptr noundef readonly captures(none), ptr noundef readonly captures(none)) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @remove(ptr noundef readonly captures(none)) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #7

declare i32 @__isoc99_fscanf(ptr noundef, ptr noundef, ...) local_unnamed_addr #5

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #9

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #10

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #8 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nounwind }
attributes #10 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #11 = { nounwind }
attributes #12 = { cold }
attributes #13 = { cold noreturn nounwind }

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
