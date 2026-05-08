; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s | FileCheck %s
; CHECK: result is 6


@.str = private constant [14 x i8] c"result is %d\0A\00", align 1

declare i32 @printf(ptr, ...)

define i32 @sum(i32 %0, ...)  {
  %2 = alloca ptr, align 8
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = va_arg ptr %2, i32
  %4 = add nsw i32 %3, %0
  %5 = va_arg ptr %2, i32
  %6 = add nsw i32 %4, %5
  call void @llvm.va_end.p0(ptr nonnull %2)
  ret i32 %6
}

define i32 @main() {
  %1 = tail call i32 (i32, ...) @sum(i32 noundef 1, i32 noundef 2, i32 noundef 3)
  %2 = tail call i32 (ptr, ...) @printf(ptr @.str, i32 noundef %1)
  ret i32 0
}