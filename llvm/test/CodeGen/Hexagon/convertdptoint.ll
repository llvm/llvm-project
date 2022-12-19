; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Check that we generate conversion from double precision floating point
; to 32-bit int value in IEEE complaint mode in V5.

; CHECK: r{{[0-9]+}} = convert_df2w(r{{[0-9]+}}:{{[0-9]+}}):chop

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  %a = alloca double, align 8
  %b = alloca double, align 8
  %c = alloca double, align 8
  store i32 0, ptr %retval
  store volatile double 1.540000e+01, ptr %a, align 8
  store volatile double 9.100000e+00, ptr %b, align 8
  %0 = load volatile double, ptr %a, align 8
  %1 = load volatile double, ptr %b, align 8
  %add = fadd double %0, %1
  store double %add, ptr %c, align 8
  %2 = load double, ptr %c, align 8
  %conv = fptosi double %2 to i32
  store i32 %conv, ptr %i, align 4
  %3 = load i32, ptr %i, align 4
  ret i32 %3
}
