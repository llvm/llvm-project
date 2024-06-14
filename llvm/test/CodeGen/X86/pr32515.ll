; RUN: llc -O0 -mtriple=x86_64-unknown -mcpu=skx -o - %s
; RUN: llc     -mtriple=x86_64-unknown -mcpu=skx -o - %s
; RUN: llc -O0 -mtriple=i686-unknown   -mcpu=skx -o - %s
; RUN: llc     -mtriple=i686-unknown   -mcpu=skx -o - %s
; REQUIRES: asserts

@var_26 = external dso_local global i16, align 2

define void @foo() #0 {
 %1 = alloca i16, align 2
 %2 = load i16, ptr @var_26, align 2
 %3 = zext i16 %2 to i32
 %4 = icmp ne i32 %3, 7
 %5 = zext i1 %4 to i16
 store i16 %5, ptr %1, align 2
 %6 = load i16, ptr @var_26, align 2
 %7 = zext i16 %6 to i32
 %8 = and i32 1, %7
 %9 = shl i32 %8, 0
 %10 = load i16, ptr @var_26, align 2
 %11 = zext i16 %10 to i32
 %12 = icmp ne i32 %11, 7
 %13 = zext i1 %12 to i32
 %14 = and i32 %9, %13
 %15 = icmp ne i32 %14, 0
 %16 = zext i1 %15 to i8
 store i8 %16, ptr undef, align 1
 unreachable
 }
