; Check that when removing arguments, existing fast math flags are preserved

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT %s < %t

; INTERESTING-LABEL: @math_callee(
define float @math_callee(float %a, float %b) {
  %add = fadd float %a, %b
  ret float %add
}

; INTERESTING-LABEL: @math_callee_decl(
declare float @math_callee_decl(float %a, float %b)

; INTERESTING-LABEL: @math_caller(
; INTERESTING: call
; INTERESTING: call

; RESULT: %call0 = call nnan nsz float @math_callee(), !fpmath !0
; RESULT: %call1 = call ninf float @math_callee_decl()
define float @math_caller(float %x) {
  %call0 = call nnan nsz float @math_callee(float %x, float 2.0), !fpmath !0
  %call1 = call ninf float @math_callee_decl(float %x, float 2.0)
  %result = fadd float %call0, %call1
  ret float %result
}

; RESULT: !0 = !{float 2.000000e+00}
!0 = !{float 2.0}
