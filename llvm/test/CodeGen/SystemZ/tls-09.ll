; RUN: llc < %s -mcpu=z196 -mtriple=s390x-linux-gnu -O0
;
; Test that a0 and a1 are copied successfully into GR32 registers.

@x = dso_local thread_local global i32 0, align 4
define i32 @fun0(i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext)  {
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = load i32, ptr @x, align 4
  store i32 %0, ptr %8, align 4
  store i32 %1, ptr %9, align 4
  store i32 %2, ptr %10, align 4
  store i32 %3, ptr %11, align 4
  store i32 %4, ptr %12, align 4
  store i32 %5, ptr %13, align 4
  store i32 %6, ptr %14, align 4
  %16 = load i32, ptr %8, align 4
  %17 = add nsw i32 %15, %16
  %18 = load i32, ptr %9, align 4
  %19 = add nsw i32 %17, %18
  %20 = load i32, ptr %10, align 4
  %21 = add nsw i32 %19, %20
  %22 = load i32, ptr %11, align 4
  %23 = add nsw i32 %21, %22
  %24 = load i32, ptr %12, align 4
  %25 = add nsw i32 %23, %24
  %26 = load i32, ptr %13, align 4
  %27 = add nsw i32 %25, %26
  %28 = load i32, ptr %14, align 4
  %29 = add nsw i32 %27, %28
  ret i32 %29
}
