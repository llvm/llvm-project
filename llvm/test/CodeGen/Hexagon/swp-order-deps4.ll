; RUN: llc -O2 -march=hexagon < %s
; REQUIRES: asserts

; Test that the final instruction ordering code does not result in infinite
; recursion (a segmentation fault). The problem is that the order heuristics
; did not properly take into account the stage in which an instruction is
; scheduled.

%0 = type { %1, %4, %9, %28 }
%1 = type { i8, [32 x %2] }
%2 = type { i8, %3, i8, i8, i16, i8, [20 x i16], [20 x i16] }
%3 = type { i16, i8 }
%4 = type { i8, [64 x %5], [64 x %5*] }
%5 = type { i8, i8, i8*, %6 }
%6 = type { %7 }
%7 = type { i8*, %3, i8, i8, i8, i8, i16, i8, i8, i8, i16, i32, i8, [3 x i8], [3 x i16], i16, i8, i16, i8, %8, i16, i8, i16 }
%8 = type { i8, i8 }
%9 = type { i8, i8, %10*, i8, [8 x %7*], i8, i8, i8, i8, i8, %7*, i8, %7*, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i32, i32, i32, i32, i32, i32, i32, i8, i8, i16, i8, void (i8)*, i8, i8, i8, i8, i8, i8 }
%10 = type { i8, i8, i8, i8, i8, %11, %12, %13, %14 }
%11 = type { i8, i16, i16 }
%12 = type { i8, i16, i8* }
%13 = type { i8, i16 }
%14 = type { %15, %20, %25 }
%15 = type { i8, i8, %16, i8, [18 x %17] }
%16 = type { i8, i16, i16 }
%17 = type { i8, i8, [10 x %3], [10 x i16], [10 x i16], [10 x i8], %18* }
%18 = type { %19, i16, i16, %19, i16 }
%19 = type { i16, i16, i16, i8 }
%20 = type { i8, i8, %21 }
%21 = type { i8*, %22, %23 }
%22 = type { %3, i8, i8, i16, i16, i16, i8, i16 }
%23 = type { [2 x %24], [4 x i8] }
%24 = type { i8, %3, i16, i16, i16, i16, %18* }
%25 = type { i8, i8, [8 x %26] }
%26 = type { i8*, %27, %24 }
%27 = type { %3, i8, i16, i16, i16 }
%28 = type { [2 x %29], [2 x i16], i8, i8*, i16, i8, i8, %31*, %32*, %33*, %33*, [3 x %34*], i8, [2 x i8], i8, i8, [2 x i8], [2 x i8], [3 x i8] }
%29 = type <{ %30, i8, [1000 x i8] }>
%30 = type { i16, i16, [2 x i32] }
%31 = type <{ i8, i8, i16, i8 }>
%32 = type <{ i16, i16, i8, i16 }>
%33 = type <{ i8, i8, i16, i16, i16, i8, i16, i16 }>
%34 = type <{ i8, i8, i16, i16, i8, i16, i8, i8, i32, i16, i16, i16 }>

@g0 = external global [2 x %0], align 8

; Function Attrs: nounwind ssp
define void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v5, %b1 ]
  %v1 = getelementptr inbounds [2 x %0], [2 x %0]* @g0, i32 0, i32 undef, i32 1, i32 1, i32 %v0
  %v2 = getelementptr inbounds [2 x %0], [2 x %0]* @g0, i32 0, i32 undef, i32 1, i32 2, i32 %v0
  store %5* %v1, %5** %v2, align 4
  %v3 = getelementptr inbounds [2 x %0], [2 x %0]* @g0, i32 0, i32 undef, i32 1, i32 1, i32 %v0, i32 3
  %v4 = bitcast %6* %v3 to %5**
  store %5* %v1, %5** %v4, align 4
  %v5 = add nuw nsw i32 %v0, 1
  %v6 = icmp eq i32 %v5, 64
  br i1 %v6, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind ssp "target-cpu"="hexagonv55" }
