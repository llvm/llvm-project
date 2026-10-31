; Test that bare i32 formal arguments (without signext/zeroext, e.g. struct
; coercions or sitofp sources) are correctly assigned to R1L/R2L/R3L under
; the XPLINK64 calling convention.
;
; Bare i32 arguments (no extend flags) must be assigned to R1L/R2L/R3L by
; the CCIfType<[i32], CCAssignToRegAndStack<[R1L,R2L,R3L],8,8>> rule in
; CC_SystemZ_XPLINK64; without that rule they fall through to the stack
; fallback, producing a load from the parameter area instead of a register
; reference.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; Bare i32 passed to sitofp.  Correct: cefbr 0,1 (R1 is the source).
; Without the i32 rule: l 0,<offset>(4) then cefbr 0,0 (stack load).
define float @sitofp_bare_i32(i32 %x) {
; CHECK-LABEL: sitofp_bare_i32 DS 0H
; CHECK: cefbr 0,1
  %r = sitofp i32 %x to float
  ret float %r
}

; Bare i32 passed to sitofp (double).  Correct: cdfbr 0,1.
define double @sitofp_bare_i32_double(i32 %x) {
; CHECK-LABEL: sitofp_bare_i32_double DS 0H
; CHECK: cdfbr 0,1
  %r = sitofp i32 %x to double
  ret double %r
}

; Two bare i32 arguments compared.  Correct: cr 1,2 (register compare).
; Without the i32 rule: two stack loads then a memory/register compare.
define i1 @cmp_bare_i32(i32 %a, i32 %b) {
; CHECK-LABEL: cmp_bare_i32 DS 0H
; CHECK: cr 1,2
  %r = icmp eq i32 %a, %b
  ret i1 %r
}
