; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

define float @fadd_zero_strict(float %x) {
; CHECK-LABEL: fadd_zero_strict:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm1, %xmm1
; CHECK-NEXT:    addss %xmm1, %xmm0
; CHECK-NEXT:    retq
  %r = fadd float %x, 0.0
  ret float %r
}

define float @fadd_negzero(float %x) {
; CHECK-LABEL: fadd_negzero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = fadd float %x, -0.0
  ret float %r
}

define float @fadd_produce_zero(float %x) {
; CHECK-LABEL: fadd_produce_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %neg = fsub nsz float 0.0, %x
  %r = fadd nnan float %neg, %x
  ret float %r
}

define float @fadd_reassociate(float %x) {
; CHECK-LABEL: fadd_reassociate:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %sum = fadd float %x, 8.0
  %r = fadd reassoc nsz float %sum, 12.0
  ret float %r
}

define float @fadd_negzero_nsz(float %x) {
; CHECK-LABEL: fadd_negzero_nsz:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = fadd nsz float %x, -0.0
  ret float %r
}

define float @fadd_zero_nsz(float %x) {
; CHECK-LABEL: fadd_zero_nsz:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = fadd nsz float %x, 0.0
  ret float %r
}

define float @fsub_zero(float %x) {
; CHECK-LABEL: fsub_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = fsub float %x, 0.0
  ret float %r
}

define float @fsub_self(float %x) {
; CHECK-LABEL: fsub_self:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = fsub nnan float %x, %x
  ret float %r
}

define float @fsub_neg_x_y(float %x, float %y) {
; CHECK-LABEL: fsub_neg_x_y:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subss %xmm0, %xmm1
; CHECK-NEXT:    movaps %xmm1, %xmm0
; CHECK-NEXT:    retq
  %neg = fsub nsz float 0.0, %x
  %r = fadd nsz float %neg, %y
  ret float %r
}

define float @fsub_neg_y(float %x, float %y) {
; CHECK-LABEL: fsub_neg_y:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %mul = fmul float %x, 5.0
  %add = fadd float %mul, %y
  %r = fsub nsz reassoc float %y, %add
  ret float %r
}

define <4 x float> @fsub_neg_y_vector(<4 x float> %x, <4 x float> %y) {
; CHECK-LABEL: fsub_neg_y_vector:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %mul = fmul <4 x float> %x, <float 5.0, float 5.0, float 5.0, float 5.0>
  %add = fadd <4 x float> %mul, %y
  %r = fsub nsz reassoc <4 x float> %y, %add
  ret <4 x float> %r
}

define <4 x float> @fsub_neg_y_vector_nonuniform(<4 x float> %x, <4 x float> %y) {
; CHECK-LABEL: fsub_neg_y_vector_nonuniform:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %mul = fmul <4 x float> %x, <float 5.0, float 6.0, float 7.0, float 8.0>
  %add = fadd <4 x float> %mul, %y
  %r = fsub nsz reassoc <4 x float> %y, %add
  ret <4 x float> %r
}

define float @fsub_neg_y_commute(float %x, float %y) {
; CHECK-LABEL: fsub_neg_y_commute:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %mul = fmul float %x, 5.0
  %add = fadd float %y, %mul
  %r = fsub nsz reassoc float %y, %add
  ret float %r
}

define <4 x float> @fsub_neg_y_commute_vector(<4 x float> %x, <4 x float> %y) {
; CHECK-LABEL: fsub_neg_y_commute_vector:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %mul = fmul <4 x float> %x, <float 5.0, float 5.0, float 5.0, float 5.0>
  %add = fadd <4 x float> %y, %mul
  %r = fsub nsz reassoc <4 x float> %y, %add
  ret <4 x float> %r
}

; Y - (X + Y) --> -X

define float @fsub_fadd_common_op_fneg(float %x, float %y) {
; CHECK-LABEL: fsub_fadd_common_op_fneg:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %a = fadd float %x, %y
  %r = fsub reassoc nsz float %y, %a
  ret float %r
}

; Y - (X + Y) --> -X

define <4 x float> @fsub_fadd_common_op_fneg_vec(<4 x float> %x, <4 x float> %y) {
; CHECK-LABEL: fsub_fadd_common_op_fneg_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %a = fadd <4 x float> %x, %y
  %r = fsub nsz reassoc <4 x float> %y, %a
  ret <4 x float> %r
}

; Y - (Y + X) --> -X
; Commute operands of the 'add'.

define float @fsub_fadd_common_op_fneg_commute(float %x, float %y) {
; CHECK-LABEL: fsub_fadd_common_op_fneg_commute:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %a = fadd float %y, %x
  %r = fsub reassoc nsz float %y, %a
  ret float %r
}

; Y - (Y + X) --> -X

define <4 x float> @fsub_fadd_common_op_fneg_commute_vec(<4 x float> %x, <4 x float> %y) {
; CHECK-LABEL: fsub_fadd_common_op_fneg_commute_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %a = fadd <4 x float> %y, %x
  %r = fsub reassoc nsz <4 x float> %y, %a
  ret <4 x float> %r
}

define float @fsub_negzero_strict(float %x) {
; CHECK-LABEL: fsub_negzero_strict:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm1, %xmm1
; CHECK-NEXT:    addss %xmm1, %xmm0
; CHECK-NEXT:    retq
  %r = fsub float %x, -0.0
  ret float %r
}

define float @fsub_negzero_nsz(float %x) {
; CHECK-LABEL: fsub_negzero_nsz:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = fsub nsz float %x, -0.0
  ret float %r
}

define <4 x float> @fsub_negzero_strict_vector(<4 x float> %x) {
; CHECK-LABEL: fsub_negzero_strict_vector:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm1, %xmm1
; CHECK-NEXT:    addps %xmm1, %xmm0
; CHECK-NEXT:    retq
  %r = fsub <4 x float> %x, <float -0.0, float -0.0, float -0.0, float -0.0>
  ret <4 x float> %r
}

define <4 x float> @fsub_negzero_nsz_vector(<4 x float> %x) {
; CHECK-LABEL: fsub_negzero_nsz_vector:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = fsub nsz <4 x float> %x, <float -0.0, float -0.0, float -0.0, float -0.0>
  ret <4 x float> %r
}

define float @fsub_zero_nsz_1(float %x) {
; CHECK-LABEL: fsub_zero_nsz_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = fsub nsz float %x, 0.0
  ret float %r
}

define float @fsub_zero_nsz_2(float %x) {
; CHECK-LABEL: fsub_zero_nsz_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %r = fsub nsz float 0.0, %x
  ret float %r
}

define float @fmul_zero(float %x) {
; CHECK-LABEL: fmul_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = fmul nnan nsz float %x, 0.0
  ret float %r
}

define float @fmul_one(float %x) {
; CHECK-LABEL: fmul_one:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = fmul float %x, 1.0
  ret float %r
}

define float @fmul_x_const_const(float %x) {
; CHECK-LABEL: fmul_x_const_const:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %mul = fmul reassoc float %x, 9.0
  %r = fmul reassoc float %mul, 4.0
  ret float %r
}
