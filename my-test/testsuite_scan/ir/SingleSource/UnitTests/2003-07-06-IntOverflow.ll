; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-07-06-IntOverflow.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-07-06-IntOverflow.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str.2 = private unnamed_addr constant [35 x i8] c"divide after overflow = %d (0x%x)\0A\00", align 1
@.str.3 = private unnamed_addr constant [49 x i8] c"divide negative value by power-of-2 = %d (0x%x)\0A\00", align 1
@.str.4 = private unnamed_addr constant [37 x i8] c"subtract after overflow = %d (0x%x)\0A\00", align 1
@str = private unnamed_addr constant [32 x i8] c"compare after overflow is FALSE\00", align 4
@str.5 = private unnamed_addr constant [31 x i8] c"compare after overflow is TRUE\00", align 4

; Function Attrs: nofree nounwind uwtable
define dso_local void @compareOvf(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = mul nsw i32 %0, %0
  %4 = mul nsw i32 %1, %1
  %5 = add nuw nsw i32 %4, %3
  %6 = icmp samesign ult i32 %5, 4194304
  %7 = select i1 %6, ptr @str.5, ptr @str
  %8 = tail call i32 @puts(ptr nonnull dereferenceable(1) %7)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @divideOvf(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = mul nsw i32 %0, %0
  %4 = mul nsw i32 %1, %1
  %5 = add nuw nsw i32 %4, %3
  %6 = sdiv i32 -2147483648, %5
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %6)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @divideNeg(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add nsw i32 %1, %0
  %4 = sub nsw i32 %0, %1
  %5 = mul nsw i32 %3, %4
  %6 = sdiv i32 %5, 262144
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %6, i32 noundef %6)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @subtractOvf(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = mul i32 %0, %0
  %4 = mul i32 %1, %1
  %5 = add i32 %4, %3
  %6 = sub i32 -2147483648, %5
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %6, i32 noundef %6)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.5)
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef -170, i32 noundef -170)
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef -16, i32 noundef -16)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef 2134900731, i32 noundef 2134900731)
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
