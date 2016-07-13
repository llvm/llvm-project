; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: {{^}}same_dynamic_index_fp_vector_type:
; CHECK: # BB#0:
; CHECK-NEXT: retq
define float @same_dynamic_index_fp_vector_type(float %val, i32 %idx) {
bb:
  %tmp0 = insertelement <4 x float> undef, float %val, i32 %idx
  %tmp1 = extractelement <4 x float> %tmp0, i32 %idx
  ret float %tmp1
}
