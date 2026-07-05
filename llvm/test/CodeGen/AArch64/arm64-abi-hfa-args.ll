; RUN: llc < %s -mtriple=arm64-none-eabi | FileCheck %s

; Over-aligned HFA argument placed on register - one element per register
define double @test_hfa_align_arg_reg([2 x double] alignstack(16) %h.coerce) local_unnamed_addr #0 {
entry:
; CHECK-LABEL: test_hfa_align_arg_reg:
; CHECK-NOT: mov
; CHECK-NOT: ld
; CHECK: ret
  %h.coerce.fca.0.extract = extractvalue [2 x double] %h.coerce, 0
  ret double %h.coerce.fca.0.extract
}

; Call with over-aligned HFA argument placed on register - one element per register
define double @test_hfa_align_call_reg() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: test_hfa_align_call_reg:
; CHECK-DAG: fmov  d0, #1.00000000
; CHECK-DAG: fmov  d1, #2.00000000
; CHECK:     bl    test_hfa_align_arg_reg
  %call = call double @test_hfa_align_arg_reg([2 x double] alignstack(16) [double 1.000000e+00, double 2.000000e+00])
  ret double %call
}

; Over-aligned HFA argument placed on stack - stack round up to alignment
define double @test_hfa_align_arg_stack(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %f, [2 x double] alignstack(16) %h.coerce) local_unnamed_addr #0 {
entry:
; CHECK-LABEL: test_hfa_align_arg_stack:
; CHECK:       ldr  d0, [sp, #16]
; CHECK-NEXT:  ret
  %h.coerce.fca.0.extract = extractvalue [2 x double] %h.coerce, 0
  ret double %h.coerce.fca.0.extract
}
