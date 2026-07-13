; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; Check that we generate conversion from double precision floating point
; to 64-bit integer value in IEEE complaint mode in V5.

; CHECK: r{{[0-9]+}}:{{[0-9]+}} = convert_df2d(r{{[0-9]+}}:{{[0-9]+}}):chop

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %i = alloca i64, align 8
  %a = alloca double, align 8
  %b = alloca double, align 8
  %c = alloca double, align 8
  store i32 0, ptr %retval
  store double 1.540000e+01, ptr %a, align 8
  store double 9.100000e+00, ptr %b, align 8
  %0 = load double, ptr %a, align 8
  %1 = load double, ptr %b, align 8
  %add = fadd double %0, %1
  store volatile double %add, ptr %c, align 8
  %2 = load volatile double, ptr %c, align 8
  %conv = fptosi double %2 to i64
  store i64 %conv, ptr %i, align 8
  %3 = load i64, ptr %i, align 8
  %conv1 = trunc i64 %3 to i32
  ret i32 %conv1
}
