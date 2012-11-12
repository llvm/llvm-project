; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Check that we generate double precision floating point multiply in V5.

; CHECK: r{{[0-9]+}}:{{[0-9]+}} = dfmpy(r{{[0-9]+}}:{{[0-9]+}}, r{{[0-9]+}}:{{[0-9]+}})

define i32 @main() nounwind {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  %c = alloca double, align 8
  store double 1.540000e+01, double* %a, align 8
  store double 9.100000e+00, double* %b, align 8
  %0 = load double* %b, align 8
  %1 = load double* %a, align 8
  %mul = fmul double %0, %1
  store double %mul, double* %c, align 8
  ret i32 0
}
