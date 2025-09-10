; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2002-05-02-ManyArguments.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2002-05-02-ManyArguments.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [20 x i8] c"%d, %d, %f, %f, %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [24 x i8] c"%d, %ld, %ld, %d, %lld\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @printfn(i32 noundef %0, i16 noundef %1, double noundef %2, float noundef %3, i8 noundef %4, i8 noundef %5, ptr noundef %6, ptr noundef %7, i32 noundef %8, i64 noundef %9) local_unnamed_addr #0 {
  %11 = sext i16 %1 to i32
  %12 = fpext float %3 to double
  %13 = sext i8 %4 to i32
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %0, i32 noundef %11, double noundef %2, double noundef %12, i32 noundef %13)
  %15 = zext i8 %5 to i32
  %16 = ptrtoint ptr %6 to i64
  %17 = ptrtoint ptr %7 to i64
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, i64 noundef %16, i64 noundef %17, i32 noundef %8, i64 noundef %9)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 12, i32 noundef 2, double noundef 1.232340e+02, double noundef 0x40933C7E20000000, i32 noundef -12)
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 23, i64 noundef 123456, i64 noundef 0, i32 noundef 1234567, i64 noundef 123124124124)
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
