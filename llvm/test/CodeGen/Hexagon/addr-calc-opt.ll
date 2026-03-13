; RUN: llc -mtriple=hexagon < %s | FileCheck %s
;
; Test whether we can produce minimal code for this complex address
; calculation.
;

; CHECK: r0 = memub(r{{[0-9]+}}<<#3+##the_global+516)

%0 = type { [3 x %1] }
%1 = type { %2, i8, i8, i8, i8, i8, [4 x i8], i8, [10 x i8], [10 x i8], [10 x i8], i8, [3 x %4], i16, i16, i16, i16, i32, i8, [4 x i8], i8, i8, i8, i8, %5, i8, i8, i8, i8, i8, i16, i8, i8, i8, i16, i16, i8, i8, [2 x i8], [2 x i8], i8, i8, i8, i8, i8, i16, i16, i8, i8, i8, i8, i8, i8, %9, i8, [6 x [2 x i8]], i16, i32, %10, [28 x i8], [4 x %17] }
%2 = type { %3 }
%3 = type { i8, i8, i8, i8, i8, i16, i16, i16, i16, i16 }
%4 = type { i16, i16 }
%5 = type { [10 x %6] }
%6 = type { [2 x %7] }
%7 = type { i8, [2 x %8] }
%8 = type { [4 x i8] }
%9 = type { i8 }
%10 = type { %11, %13 }
%11 = type { [2 x [2 x i8]], [2 x [8 x %12]], [6 x i16], [6 x i16] }
%12 = type { i8, i8 }
%13 = type { [4 x %12], [4 x %12], [2 x [4 x %14]], [6 x i16] }
%14 = type { %15, %16 }
%15 = type { i8, i8 }
%16 = type { i8, i8 }
%17 = type { i8, i8, ptr, i16, i16, i16, i64, i32, i32, %18, i8, %21, i8, [2 x i16], i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i16, i16, i16, i8, i8, i8, i16, i16, [2 x i16], i16, [2 x i32], [2 x i16], [2 x i16], i8, i8, [6 x %23], i8, i8, i8, %24, %25, %26, %28 }
%18 = type { %19, [10 x %20] }
%19 = type { i32 }
%20 = type { [2 x i8], [2 x i8], i8, i8, i8, i8 }
%21 = type { i8, i8, i8, [8 x %22] }
%22 = type { i8, i8, i8, i32 }
%23 = type { i32, i16, i16, [2 x i16], [2 x i16], [2 x i16], i32 }
%24 = type { [2 x i32], [2 x ptr], [2 x ptr], [2 x ptr], [2 x i32], [2 x i32], i32 }
%25 = type { [2 x i32], [2 x i32], [2 x i32] }
%26 = type { i8, i8, i8, i16, i16, %27, i32, i32, i32, i16 }
%27 = type { i64 }
%28 = type { %29, %31, [24 x i8] }
%29 = type { [2 x %30], [16 x i32] }
%30 = type { [16 x i32], [8 x i32], [16 x i32], [64 x i32], [2 x i32], i64, i32, i32, i32, i32 }
%31 = type { [2 x %32] }
%32 = type { [4 x %33], i32 }
%33 = type { i32, i32 }

@the_global = external global %0

; Function Attrs: nounwind optsize readonly ssp
define zeroext i8 @myFun(i8 zeroext, i8 zeroext) {
  %3 = zext i8 %1 to i32
  %4 = zext i8 %0 to i32
  %5 = getelementptr inbounds %0, ptr @the_global, i32 0, i32 0, i32 %4, i32 60, i32 0, i32 9, i32 1, i32 %3, i32 0, i32 0
  %6 = load i8, ptr %5, align 4
  ret i8 %6
}

