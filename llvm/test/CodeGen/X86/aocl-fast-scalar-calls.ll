; Verify that, under fast-math at -O3 with -fast-library=AMDLIBM, scalar math
; library calls are rewritten to their AMD AOCL fast-call equivalents on X86,
; and that they are left untouched without the option.

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O3 -fast-library=AMDLIBM < %s \
; RUN:   | FileCheck %s --check-prefix=AMD
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O3 < %s \
; RUN:   | FileCheck %s --check-prefix=STD

declare double @tan(double)
declare double @exp(double)
declare double @log(double)

define double @call_tan(double %x) #0 {
entry:
  %r = call double @tan(double %x)
  %a = fadd double %r, %x
  ret double %a
}

; AMD-LABEL: call_tan:
; AMD: callq{{.*}}amd_fasttan
; STD-LABEL: call_tan:
; STD: callq{{.*}}tan

define double @call_exp(double %x) #0 {
entry:
  %r = call double @exp(double %x)
  %a = fadd double %r, %x
  ret double %a
}

; AMD-LABEL: call_exp:
; AMD: callq{{.*}}amd_fastexp
; STD-LABEL: call_exp:
; STD: callq{{.*}}exp

define double @call_log(double %x) #0 {
entry:
  %r = call double @log(double %x)
  %a = fadd double %r, %x
  ret double %a
}

; AMD-LABEL: call_log:
; AMD: callq{{.*}}amd_fastlog
; STD-LABEL: call_log:
; STD: callq{{.*}}log

; Without the fast-math attributes the call must not be rewritten even when the
; AMD scalar library is selected.
define double @call_tan_no_fastmath(double %x) {
entry:
  %r = call double @tan(double %x)
  %a = fadd double %r, %x
  ret double %a
}

; AMD-LABEL: call_tan_no_fastmath:
; AMD-NOT: amd_fasttan
; AMD: callq{{.*}}tan

attributes #0 = { "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" }
