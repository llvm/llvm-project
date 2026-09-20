; RUN: not llc -mtriple=x86_64-- -mattr=-sse,-x87 %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSSE --implicit-check-not=error:
; RUN: not llc -mtriple=x86_64-- -mattr=-sse2,-x87 %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK --implicit-check-not=error:

; On x86-64 the hard-float ABI returns scalar floating-point values in SSE
; registers. When the required SSE feature and x87 are both disabled the type
; is softened and the ABI silently degrades to soft-float. Diagnose the
; mismatch instead. See https://github.com/llvm/llvm-project/issues/111406.

; NOSSE: error: {{.*}} in function ret_float float (float): SSE register return with SSE disabled
define float @ret_float(float %a) {
  ret float %a
}

; CHECK: error: {{.*}} in function ret_double double (double): SSE2 register return with SSE2 disabled
define double @ret_double(double %a) {
  ret double %a
}

; CHECK: error: {{.*}} in function ret_half half (half): SSE2 register return with SSE2 disabled
define half @ret_half(half %a) {
  ret half %a
}

; A call whose result is returned in an SSE register is diagnosed as well.
; NOSSE: error: {{.*}} in function call_float {{.*}}: SSE register return with SSE disabled
define void @call_float() {
  %r = call float @extern_float()
  ret void
}

; An explicit soft-float ABI has no SSE-register requirement to violate.
define float @soft_float_ok(float %a) "target-features"="+soft-float,-sse,-x87" {
  ret float %a
}

; Integer-only functions are fine.
define i32 @no_fp(i32 %a) {
  ret i32 %a
}

declare float @extern_float()
