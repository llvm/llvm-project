; Check that we can enable/disable NoInfsFPMath and NoNaNsInFPMath via function
; attributes.  An attribute on one function should not magically apply to the
; next one.

; RUN: llc < %s -mtriple=powerpc64-unknown-unknown -mcpu=pwr7 -mattr=-vsx \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=SAFE

; RUN: llc < %s -mtriple=powerpc64-unknown-unknown -mcpu=pwr7 -mattr=-vsx \
; RUN:   -enable-no-infs-fp-math -enable-no-nans-fp-math \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=UNSAFE

; The fcmp+select in these functions should be converted to a fsel instruction
; when both NoInfsFPMath and NoNaNsInFPMath are enabled.

; CHECK-LABEL: default0:
define double @default0(double %a, double %y, double %z) {
entry:
; SAFE-NOT:  fsel
; UNSAFE:    fsel
  %cmp = fcmp ult double %a, 0.000000e+00
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y
}

; CHECK-LABEL: unsafe_math_off:
define double @unsafe_math_off(double %a, double %y, double %z) #0 #2 {
entry:
; SAFE-NOT:   fsel
; UNSAFE-NOT: fsel
  %cmp = fcmp ult double %a, 0.000000e+00
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y
}

; CHECK-LABEL: default1:
define double @default1(double %a, double %y, double %z) {
; SAFE-NOT:  fsel
; UNSAFE:    fsel
  %cmp = fcmp ult double %a, 0.000000e+00
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y
}

; CHECK-LABEL: unsafe_math_on:
define double @unsafe_math_on(double %a, double %y, double %z) #1 #3 {
entry:
; SAFE-NOT:   fsel
; UNSAFE-NOT: fsel
  %cmp = fcmp ult double %a, 0.000000e+00
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y
}

; CHECK-LABEL: default2:
define double @default2(double %a, double %y, double %z) {
; SAFE-NOT:  fsel
; UNSAFE:    fsel
  %cmp = fcmp ult double %a, 0.000000e+00
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y
}

attributes #0 = { "no-infs-fp-math"="false" }
attributes #1 = { "no-nans-fp-math"="false" }

attributes #2 = { "no-infs-fp-math"="false" }
attributes #3 = { "no-infs-fp-math"="true" }
