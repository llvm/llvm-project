; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Check that we generate double precision floating point multiply in V5.

; CHECK: call __hexagon_muldf3

define i32 @main() nounwind {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  %c = alloca double, align 8
  store volatile double 1.540000e+01, ptr %a, align 8
  store volatile double 9.100000e+00, ptr %b, align 8
  %0 = load volatile double, ptr %b, align 8
  %1 = load volatile double, ptr %a, align 8
  %mul = fmul double %0, %1
  store double %mul, ptr %c, align 8
  ret i32 0
}
