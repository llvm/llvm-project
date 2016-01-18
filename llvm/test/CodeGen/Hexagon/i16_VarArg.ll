; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: call __hexagon_{{[A-Z_a-z0-9]+}}

@a_str = internal constant [8 x i8] c"a = %f\0A\00"
@b_str = internal constant [8 x i8] c"b = %f\0A\00"
@add_str = internal constant [12 x i8] c"a + b = %f\0A\00"
@sub_str = internal constant [12 x i8] c"a - b = %f\0A\00"
@mul_str = internal constant [12 x i8] c"a * b = %f\0A\00"
@div_str = internal constant [12 x i8] c"b / a = %f\0A\00"
@rem_str = internal constant [13 x i8] c"b %% a = %f\0A\00"
@lt_str = internal constant [12 x i8] c"a < b = %d\0A\00"
@le_str = internal constant [13 x i8] c"a <= b = %d\0A\00"
@gt_str = internal constant [12 x i8] c"a > b = %d\0A\00"
@ge_str = internal constant [13 x i8] c"a >= b = %d\0A\00"
@eq_str = internal constant [13 x i8] c"a == b = %d\0A\00"
@ne_str = internal constant [13 x i8] c"a != b = %d\0A\00"
@A = global double 2.000000e+00
@B = global double 5.000000e+00

declare i32 @printf(i8*, ...)

define i32 @main() {
        %a = load double, double* @A
        %b = load double, double* @B
        %lt_r = fcmp olt double %a, %b
        %le_r = fcmp ole double %a, %b
        %gt_r = fcmp ogt double %a, %b
        %ge_r = fcmp oge double %a, %b
        %eq_r = fcmp oeq double %a, %b
        %ne_r = fcmp une double %a, %b
        %val1 = zext i1 %lt_r to i16
        %lt_s = getelementptr [12 x i8], [12 x i8]* @lt_str, i64 0, i64 0
        %le_s = getelementptr [13 x i8], [13 x i8]* @le_str, i64 0, i64 0
        %gt_s = getelementptr [12 x i8], [12 x i8]* @gt_str, i64 0, i64 0
        %ge_s = getelementptr [13 x i8], [13 x i8]* @ge_str, i64 0, i64 0
        %eq_s = getelementptr [13 x i8], [13 x i8]* @eq_str, i64 0, i64 0
        %ne_s = getelementptr [13 x i8], [13 x i8]* @ne_str, i64 0, i64 0
        call i32 (i8*, ...) @printf( i8* %lt_s, i16 %val1 )
        ret i32 0
}
