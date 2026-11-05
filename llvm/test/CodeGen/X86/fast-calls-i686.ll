; Fast library call lowering applies to 32-bit X86 (i686) as well as x86_64.

; RUN: llc -mtriple=i686-unknown-linux-gnu -O3 -fast-library=AMDLIBM < %s \
; RUN:   | FileCheck %s --check-prefix=AMD
; RUN: llc -mtriple=i686-unknown-linux-gnu -O3 < %s \
; RUN:   | FileCheck %s --check-prefix=STD

declare double @tan(double)
declare float @expf(float)

define double @call_tan(double %x) #0 {
  %r = call double @tan(double %x)
  %a = fadd double %r, %x
  ret double %a
}
; AMD-LABEL: call_tan:
; AMD: calll{{.*}}amd_fasttan
; STD-LABEL: call_tan:
; STD: calll{{.*}}tan

define float @call_expf(float %x) #0 {
  %r = call float @expf(float %x)
  %a = fadd float %r, %x
  ret float %a
}
; AMD-LABEL: call_expf:
; AMD: calll{{.*}}amd_fastexpf
; STD-LABEL: call_expf:
; STD: calll{{.*}}expf

attributes #0 = { "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" }
