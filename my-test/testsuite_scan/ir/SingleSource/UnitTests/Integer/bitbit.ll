; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/bitbit.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/bitbit.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [15 x i8] c"i_result = %x\0A\00", align 1
@str.7 = private unnamed_addr constant [3 x i8] c"ok\00", align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @my_test() local_unnamed_addr #0 {
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef -2097152)
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1048576)
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %4 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %5 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %6 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1)
  %8 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %9 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef -2097152)
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1048576)
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %4 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %5 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %6 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1)
  %8 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %9 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
