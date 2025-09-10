; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SetjmpLongjmp/C/FarJump.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SetjmpLongjmp/C/FarJump.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__jmp_buf_tag = type { [22 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

@str = private unnamed_addr constant [12 x i8] c"Inside quux\00", align 4
@str.9 = private unnamed_addr constant [26 x i8] c"Longjmping from quux: 927\00", align 4
@str.10 = private unnamed_addr constant [11 x i8] c"Inside qux\00", align 4
@str.11 = private unnamed_addr constant [32 x i8] c"Error: Shouldn't be here in qux\00", align 4
@str.12 = private unnamed_addr constant [11 x i8] c"Inside baz\00", align 4
@str.13 = private unnamed_addr constant [32 x i8] c"Error: Shouldn't be here in baz\00", align 4
@str.14 = private unnamed_addr constant [11 x i8] c"Inside bar\00", align 4
@str.15 = private unnamed_addr constant [11 x i8] c"Inside foo\00", align 4
@str.16 = private unnamed_addr constant [32 x i8] c"Returning from longjmp into foo\00", align 4

; Function Attrs: noreturn nounwind uwtable
define dso_local void @quux(ptr noundef %0) local_unnamed_addr #0 {
  %2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.9)
  tail call void @longjmp(ptr noundef %0, i32 noundef 927) #6
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @longjmp(ptr noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local void @qux(ptr noundef %0) local_unnamed_addr #2 {
  %2 = alloca [1 x %struct.__jmp_buf_tag], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  %3 = call i32 @puts(ptr nonnull dereferenceable(1) @str.10)
  %4 = call i32 @_setjmp(ptr noundef nonnull %2) #8
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %8 = call i32 @puts(ptr nonnull dereferenceable(1) @str.9)
  call void @longjmp(ptr noundef %0, i32 noundef 927) #6
  unreachable

9:                                                ; preds = %1
  %10 = call i32 @puts(ptr nonnull dereferenceable(1) @str.11)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: nounwind returns_twice
declare i32 @_setjmp(ptr noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: nounwind uwtable
define dso_local void @baz(ptr noundef %0) local_unnamed_addr #2 {
  %2 = alloca [1 x %struct.__jmp_buf_tag], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  %3 = call i32 @puts(ptr nonnull dereferenceable(1) @str.12)
  %4 = call i32 @_setjmp(ptr noundef nonnull %2) #8
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %1
  call void @qux(ptr noundef %0)
  br label %9

7:                                                ; preds = %1
  %8 = call i32 @puts(ptr nonnull dereferenceable(1) @str.13)
  br label %9

9:                                                ; preds = %7, %6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @bar(ptr noundef %0) local_unnamed_addr #2 {
  %2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  tail call void @baz(ptr noundef %0)
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @foo() local_unnamed_addr #2 {
  %1 = alloca [1 x %struct.__jmp_buf_tag], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %2 = call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  %3 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %7

5:                                                ; preds = %0
  %6 = call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  call void @baz(ptr noundef nonnull %1)
  br label %9

7:                                                ; preds = %0
  %8 = call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  br label %9

9:                                                ; preds = %7, %5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  tail call void @foo()
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #5

attributes #0 = { noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nounwind returns_twice "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind }
attributes #8 = { nounwind returns_twice }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
