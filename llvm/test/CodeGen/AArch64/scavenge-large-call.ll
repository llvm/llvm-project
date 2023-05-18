; RUN: llc -mtriple=arm64_32-apple-watchos %s -o - | FileCheck %s

; CHECK-LABEL: caller:
; CHECK: add {{x[0-9]+}}, sp,

define void @caller(ptr %0, i16 %1, i16 %2, i8 %3, double %4, i16 %5, i8 %6, ptr %7, double %8, i32 %9, ptr %10, double %11, double %12, [2 x i64] %13, [2 x i64] %14, [2 x i64] %15, double %16, double %17, [2 x i64] %18, [2 x i64] %19, i16 %20, i32 %21, double %22, i8 %23, [2 x i64] %24, [2 x i64] %25, [2 x i64] %26, i8 %27, i16 %28, i16 %29, i16 %30, i32 %31, [2 x i64] %32, [2 x i64] %33, [2 x i64] %34, [2 x i64] %35, [2 x i64] %36, i32 %37, i32 %38) {
  tail call void @callee(ptr %0, i16 %1, i16 %2, i8 %3, double 0.000000e+00, i16 %5, i8 %6, ptr %7, double 0.000000e+00, i32 %9, ptr %10, double 0.000000e+00, double 0.000000e+00, [2 x i64] %13, [2 x i64] %14, [2 x i64] %15, double 0.000000e+00, double 0.000000e+00, [2 x i64] %18, [2 x i64] %19, i16 %20, i32 %21, double 0.000000e+00, i8 %23, [2 x i64] %24, [2 x i64] %25, [2 x i64] zeroinitializer, i8 %27, i16 0, i16 0, i16 %28, i32 0, [2 x i64] zeroinitializer, [2 x i64] zeroinitializer, [2 x i64] zeroinitializer, [2 x i64] %35, [2 x i64] %36, i32 0, i32 0)
  ret void
}

declare void @callee(ptr, i16, i16, i8, double, i16, i8, ptr, double, i32, ptr, double, double, [2 x i64], [2 x i64], [2 x i64], double, double, [2 x i64], [2 x i64], i16, i32, double, i8, [2 x i64], [2 x i64], [2 x i64], i8, i16, i16, i16, i32, [2 x i64], [2 x i64], [2 x i64], [2 x i64], [2 x i64], i32, i32)
