; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: dfcmp

@g0 = internal constant [12 x i8] c"a < b = %d\0A\00"
@g1 = internal constant [13 x i8] c"a <= b = %d\0A\00"
@g2 = internal constant [12 x i8] c"a > b = %d\0A\00"
@g3 = internal constant [13 x i8] c"a >= b = %d\0A\00"
@g4 = internal constant [13 x i8] c"a == b = %d\0A\00"
@g5 = internal constant [13 x i8] c"a != b = %d\0A\00"
@g6 = global double 2.000000e+00
@g7 = global double 5.000000e+00

declare i32 @f0(ptr, ...) #0

define i32 @f1() #0 {
b0:
  %v0 = load double, ptr @g6
  %v1 = load double, ptr @g7
  %v2 = fcmp olt double %v0, %v1
  %v3 = fcmp ole double %v0, %v1
  %v4 = fcmp ogt double %v0, %v1
  %v5 = fcmp oge double %v0, %v1
  %v6 = fcmp oeq double %v0, %v1
  %v7 = fcmp une double %v0, %v1
  %v8 = zext i1 %v2 to i8
  %v15 = call i32 (ptr, ...) @f0(ptr @g0, i8 %v8)
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
