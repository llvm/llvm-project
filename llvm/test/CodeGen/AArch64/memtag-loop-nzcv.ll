; RUN: llc -O2 -print-after-isel -mtriple=aarch64-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK

; This test function includes a 256-byte buffer. We expect it to require its
; MTE tags to be set to a useful value on entry, and cleared again on exit. At
; the time of writing this test, the pseudo-instructions chosen are
; STGloop_wback and STGloop respectively, but if different pseudos are selected
; in future, that's not a problem. The important thing is that both should
; include that implicit-def of $nzcv, because these pseudo-instructions will
; expand into loops that use the flags for their termination tests.

; CHECK: STGloop_wback 256, {{.*}}, implicit-def dead $nzcv
; CHECK: STGloop       256, {{.*}}, implicit-def dead $nzcv

define i32 @foo(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca [256 x i8], align 1
  %4 = alloca i64, align 8
  %5 = alloca i32, align 4
  %6 = alloca i64, align 8
  store i32 %0, ptr %2, align 4
  %7 = load i32, ptr %2, align 4
  %8 = getelementptr inbounds [256 x i8], ptr %3, i64 0, i64 0
  %9 = call i64 @read(i32 noundef %7, ptr noundef %8, i64 noundef 256)
  store i64 %9, ptr %4, align 8
  store i32 0, ptr %5, align 4
  store i64 0, ptr %6, align 8
  br label %10

10:                                               ; preds = %21, %1
  %11 = load i64, ptr %6, align 8
  %12 = load i64, ptr %4, align 8
  %13 = icmp ult i64 %11, %12
  br i1 %13, label %14, label %24

14:                                               ; preds = %10
  %15 = load i64, ptr %6, align 8
  %16 = getelementptr inbounds [256 x i8], ptr %3, i64 0, i64 %15
  %17 = load i8, ptr %16, align 1
  %18 = zext i8 %17 to i32
  %19 = load i32, ptr %5, align 4
  %20 = add nsw i32 %19, %18
  store i32 %20, ptr %5, align 4
  br label %21

21:                                               ; preds = %14
  %22 = load i64, ptr %6, align 8
  %23 = add i64 %22, 1
  store i64 %23, ptr %6, align 8
  br label %10

24:                                               ; preds = %10
  %25 = load i32, ptr %5, align 4
  %26 = srem i32 %25, 251
  ret i32 %26
}

declare i64 @read(i32 noundef, ptr noundef, i64 noundef)

attributes #0 = { sanitize_memtag "target-features"="+mte" }
