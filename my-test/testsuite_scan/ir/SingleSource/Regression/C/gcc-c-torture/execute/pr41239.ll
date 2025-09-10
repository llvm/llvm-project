; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr41239.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr41239.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"foo\00", align 1
@__func__.test = private unnamed_addr constant [5 x i8] c"test\00", align 1
@.str.1 = private unnamed_addr constant [17 x i8] c"division by zero\00", align 1
@__const.main.s = private unnamed_addr constant { i16, [6 x i8], [2 x i64] } { i16 2, [6 x i8] zeroinitializer, [2 x i64] [i64 5, i64 0] }, align 8

; Function Attrs: nounwind uwtable
define dso_local range(i64 -2147483648, 2147483649) i64 @test(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load i64, ptr %2, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %5 = load i64, ptr %4, align 8, !tbaa !6
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %7, label %13

7:                                                ; preds = %1
  %8 = tail call i8 @fn1(i32 noundef 20, ptr noundef nonnull @.str, i32 noundef 924, ptr noundef nonnull @__func__.test, ptr noundef null)
  %9 = icmp eq i8 %8, 0
  br i1 %9, label %13, label %10

10:                                               ; preds = %7
  %11 = tail call i32 @fn3(i32 noundef 33816706)
  %12 = tail call i32 (ptr, ...) @fn4(ptr noundef nonnull @.str.1)
  tail call void (i32, ...) @fn2(i32 noundef %11, i32 noundef %12)
  br label %13

13:                                               ; preds = %10, %7, %1
  %14 = shl i64 %3, 32
  %15 = ashr exact i64 %14, 32
  %16 = sdiv i64 %15, %5
  ret i64 %16
}

; Function Attrs: noinline nounwind uwtable
define dso_local i8 @fn1(i32 noundef %0, ptr noundef %1, i32 noundef %2, ptr noundef %3, ptr noundef %4) local_unnamed_addr #1 {
  tail call void asm sideeffect "", "r,r,~{memory}"(ptr %3, ptr %4) #5, !srcloc !10
  %6 = tail call i32 asm sideeffect "", "=r,r,r,0,~{memory}"(ptr %1, i32 %2, i32 %0) #5, !srcloc !11
  %7 = trunc i32 %6 to i8
  ret i8 %7
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn2(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = tail call i32 asm sideeffect "", "=r,0,~{memory}"(i32 %0) #5, !srcloc !12
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @exit(i32 noundef 0) #6
  unreachable

5:                                                ; preds = %1
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @fn3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = tail call i32 asm sideeffect "", "=r,0,~{memory}"(i32 %0) #5, !srcloc !13
  ret i32 %2
}

; Function Attrs: noinline nounwind uwtable
define dso_local range(i32 0, 256) i32 @fn4(ptr noundef %0, ...) local_unnamed_addr #1 {
  %2 = tail call ptr asm sideeffect "", "=r,0,~{memory}"(ptr %0) #5, !srcloc !14
  %3 = load i8, ptr %2, align 1, !tbaa !15
  %4 = zext i8 %3 to i32
  ret i32 %4
}

; Function Attrs: cold noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = tail call i64 @test(ptr noundef nonnull @__const.main.s)
  tail call void @abort() #6
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #4

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{i64 805}
!11 = !{i64 858}
!12 = !{i64 1207}
!13 = !{i64 979}
!14 = !{i64 1096}
!15 = !{!8, !8, i64 0}
