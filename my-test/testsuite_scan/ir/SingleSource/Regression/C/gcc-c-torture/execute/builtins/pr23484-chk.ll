; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/pr23484-chk.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/pr23484-chk.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@chk_calls = external global i32, align 4
@data = internal global [8 x i8] c"ABCDEFG\00", align 1
@l1 = dso_local local_unnamed_addr global i32 0, align 4
@.str.3 = private unnamed_addr constant [3 x i8] c"%d\00", align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @test1() local_unnamed_addr #0 {
  %1 = alloca [8 x i8], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  store volatile i32 0, ptr @chk_calls, align 4, !tbaa !6
  store i64 5280832617179597129, ptr %1, align 8
  %2 = load i32, ptr @l1, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 0
  %4 = select i1 %3, i64 4, i64 8
  %5 = call ptr @__memcpy_chk(ptr noundef nonnull %1, ptr noundef nonnull @data, i64 noundef %4, i64 noundef 8) #7
  %6 = icmp eq ptr %5, %1
  %7 = load i64, ptr %1, align 8
  %8 = icmp eq i64 %7, 5280832617095316033
  %9 = select i1 %6, i1 %8, i1 false
  br i1 %9, label %11, label %10

10:                                               ; preds = %0
  call void @abort() #8
  unreachable

11:                                               ; preds = %0
  store i64 5353172790017673802, ptr %1, align 8
  %12 = call ptr @__mempcpy_chk(ptr noundef nonnull %1, ptr noundef nonnull @data, i64 noundef %4, i64 noundef 8) #7
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %14 = icmp eq ptr %12, %13
  %15 = load i64, ptr %1, align 8
  %16 = icmp eq i64 %15, 5353172789916549697
  %17 = select i1 %14, i1 %16, i1 false
  br i1 %17, label %19, label %18

18:                                               ; preds = %11
  call void @abort() #8
  unreachable

19:                                               ; preds = %11
  store i64 5425512962855750475, ptr %1, align 8
  %20 = load i32, ptr @l1, align 4, !tbaa !6
  %21 = icmp eq i32 %20, 0
  %22 = select i1 %21, i64 4, i64 8
  %23 = call ptr @__memmove_chk(ptr noundef nonnull %1, ptr noundef nonnull @data, i64 noundef %22, i64 noundef 8) #7
  %24 = icmp eq ptr %23, %1
  %25 = load i64, ptr %1, align 8
  %26 = icmp eq i64 %25, 5425512962737783361
  %27 = select i1 %24, i1 %26, i1 false
  br i1 %27, label %29, label %28

28:                                               ; preds = %19
  call void @abort() #8
  unreachable

29:                                               ; preds = %19
  store i64 5497853135693827148, ptr %1, align 8
  %30 = load i32, ptr @l1, align 4, !tbaa !6
  %31 = icmp eq i32 %30, 0
  %32 = select i1 %31, i64 4, i64 8
  %33 = add nsw i32 %30, 65536
  %34 = call i32 (ptr, i64, i32, i64, ptr, ...) @__snprintf_chk(ptr noundef nonnull %1, i64 noundef %32, i32 noundef 0, i64 noundef 8, ptr noundef nonnull @.str.3, i32 noundef %33) #7
  %35 = icmp eq i32 %34, 5
  %36 = load i64, ptr %1, align 8
  %37 = icmp eq i64 %36, 5497853134417245494
  %38 = select i1 %35, i1 %37, i1 false
  br i1 %38, label %40, label %39

39:                                               ; preds = %29
  call void @abort() #8
  unreachable

40:                                               ; preds = %29
  %41 = load volatile i32, ptr @chk_calls, align 4, !tbaa !6
  %42 = icmp eq i32 %41, 0
  br i1 %42, label %44, label %43

43:                                               ; preds = %40
  call void @abort() #8
  unreachable

44:                                               ; preds = %40
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nounwind memory(argmem: readwrite)
declare ptr @__memcpy_chk(ptr noalias noundef writeonly, ptr noalias noundef readonly captures(none), i64 noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare ptr @__mempcpy_chk(ptr noundef, ptr noundef, i64 noundef, i64 noundef) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare ptr @__memmove_chk(ptr noundef, ptr noundef, i64 noundef, i64 noundef) local_unnamed_addr #4

; Function Attrs: nofree
declare i32 @__snprintf_chk(ptr noundef, i64 noundef, i32 noundef, i64 noundef, ptr noundef, ...) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #6 {
  %1 = load i32, ptr @l1, align 4, !tbaa !6
  %2 = tail call i32 asm "", "=r,0"(i32 %1) #9, !srcloc !10
  store i32 %2, ptr @l1, align 4, !tbaa !6
  tail call void @test1()
  ret void
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nounwind memory(argmem: readwrite) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }
attributes #9 = { nounwind memory(none) }

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
!10 = !{i64 1596}
