; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 < %s | FileCheck %s
; https://llvm.org/bugs/show_bug.cgi?id=28444

; extract_vector_elt is allowed to have a different result type than
; the vector scalar type.
; This uses both
;  i8 = extract_vector_elt v1i1, Constant:i64<0>
;  i1 = extract_vector_elt v1i1, Constant:i64<0>


; CHECK-LABEL: {{^}}extractelt_mismatch_vector_element_type:
; CHECK: movb $1, %al
; CHECK: movb %al
; CHECK: movb %al
define void @extractelt_mismatch_vector_element_type(i32 %arg) {
bb:
  %tmp = icmp ult i32 %arg, 0
  %tmp2 = insertelement <1 x i1> undef, i1 true, i32 0
  %tmp3 = select i1 %tmp, <1 x i1> undef, <1 x i1> %tmp2
  %tmp6 = extractelement <1 x i1> %tmp3, i32 0
  br label %bb1

bb1:
  store volatile <1 x i1> %tmp3, <1 x i1>* undef
  store volatile i1 %tmp6, i1* undef
  ret void
}
