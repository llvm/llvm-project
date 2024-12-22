; RUN: opt -S -passes=instsimplify %s | FileCheck %s

define { float, float } @sincos_zero() {
; CHECK-LABEL: define { float, float } @sincos_zero() {
; CHECK-NEXT:    ret { float, float } { float 0.000000e+00, float 1.000000e+00 }
;
  %ret = call { float, float } @llvm.sincos.f32(float zeroinitializer)
  ret { float, float } %ret
}

define { float, float } @sincos_neg_zero() {
; CHECK-LABEL: define { float, float } @sincos_neg_zero() {
; CHECK-NEXT:    ret { float, float } { float -0.000000e+00, float 1.000000e+00 }
;
  %ret = call { float, float } @llvm.sincos.f32(float -0.0)
  ret { float, float } %ret
}

define { float, float } @sincos_one() {
; CHECK-LABEL: define { float, float } @sincos_one() {
; CHECK-NEXT:    ret { float, float } { float [[$SIN_ONE:.+]], float [[$COS_ONE:.+]] }
;
  %ret = call { float, float } @llvm.sincos.f32(float 1.0)
  ret { float, float } %ret
}

define { float, float } @sincos_two() {
; CHECK-LABEL: define { float, float } @sincos_two() {
; CHECK-NEXT:    ret { float, float } { float [[$SIN_TWO:.+]], float [[$COS_TWO:.+]] }
;
  %ret = call { float, float } @llvm.sincos.f32(float 2.0)
  ret { float, float } %ret
}

define { <2 x float>, <2 x float> } @sincos_vector() {
; CHECK-LABEL: define { <2 x float>, <2 x float> } @sincos_vector() {
; CHECK-NEXT:    ret { <2 x float>, <2 x float> } { <2 x float> <float [[$SIN_ONE]], float [[$SIN_TWO]]>, <2 x float> <float [[$COS_ONE]], float [[$COS_TWO]]> }
;
  %ret = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> <float 1.0, float 2.0>)
  ret { <2 x float>, <2 x float> } %ret
}

define { <2 x float>, <2 x float> } @sincos_zero_vector() {
; CHECK-LABEL: define { <2 x float>, <2 x float> } @sincos_zero_vector() {
; CHECK-NEXT:    ret { <2 x float>, <2 x float> } { <2 x float> zeroinitializer, <2 x float> splat (float 1.000000e+00) }
;
  %ret = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> zeroinitializer)
  ret { <2 x float>, <2 x float> } %ret
}

define { float, float } @sincos_poison() {
; CHECK-LABEL: define { float, float } @sincos_poison() {
; CHECK-NEXT:    [[RET:%.*]] = call { float, float } @llvm.sincos.f32(float poison)
; CHECK-NEXT:    ret { float, float } [[RET]]
;
  %ret = call { float, float } @llvm.sincos.f32(float poison)
  ret { float, float } %ret
}

define { <2 x float>, <2 x float> } @sincos_poison_vector() {
; CHECK-LABEL: define { <2 x float>, <2 x float> } @sincos_poison_vector() {
; CHECK-NEXT:    [[RET:%.*]] = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> poison)
; CHECK-NEXT:    ret { <2 x float>, <2 x float> } [[RET]]
;
  %ret = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> poison)
  ret { <2 x float>, <2 x float> } %ret
}

define { <vscale x 2 x float>, <vscale x 2 x float> } @sincos_poison_scalable_vector() {
; CHECK-LABEL: define { <vscale x 2 x float>, <vscale x 2 x float> } @sincos_poison_scalable_vector() {
; CHECK-NEXT:    [[RET:%.*]] = call { <vscale x 2 x float>, <vscale x 2 x float> } @llvm.sincos.nxv2f32(<vscale x 2 x float> poison)
; CHECK-NEXT:    ret { <vscale x 2 x float>, <vscale x 2 x float> } [[RET]]
;
  %ret = call { <vscale x 2 x float>, <vscale x 2 x float> } @llvm.sincos.nxv2f32(<vscale x 2 x float> poison)
  ret { <vscale x 2 x float>, <vscale x 2 x float> } %ret
}

define { float, float } @sincos_undef() {
; CHECK-LABEL: define { float, float } @sincos_undef() {
; CHECK-NEXT:    [[RET:%.*]] = call { float, float } @llvm.sincos.f32(float undef)
; CHECK-NEXT:    ret { float, float } [[RET]]
;
  %ret = call { float, float } @llvm.sincos.f32(float undef)
  ret { float, float } %ret
}

define { <2 x float>, <2 x float> } @sincos_undef_vector() {
; CHECK-LABEL: define { <2 x float>, <2 x float> } @sincos_undef_vector() {
; CHECK-NEXT:    [[RET:%.*]] = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> undef)
; CHECK-NEXT:    ret { <2 x float>, <2 x float> } [[RET]]
;
  %ret = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> undef)
  ret { <2 x float>, <2 x float> } %ret
}

define { <vscale x 2 x float>, <vscale x 2 x float> } @sincos_undef_scalable_vector() {
; CHECK-LABEL: define { <vscale x 2 x float>, <vscale x 2 x float> } @sincos_undef_scalable_vector() {
; CHECK-NEXT:    [[RET:%.*]] = call { <vscale x 2 x float>, <vscale x 2 x float> } @llvm.sincos.nxv2f32(<vscale x 2 x float> undef)
; CHECK-NEXT:    ret { <vscale x 2 x float>, <vscale x 2 x float> } [[RET]]
;
  %ret = call { <vscale x 2 x float>, <vscale x 2 x float> } @llvm.sincos.nxv2f32(<vscale x 2 x float> undef)
  ret { <vscale x 2 x float>, <vscale x 2 x float> } %ret
}

define { <vscale x 2 x float>, <vscale x 2 x float> } @sincos_zero_scalable_vector() {
; CHECK-LABEL: define { <vscale x 2 x float>, <vscale x 2 x float> } @sincos_zero_scalable_vector() {
; CHECK-NEXT:    [[RET:%.*]] = call { <vscale x 2 x float>, <vscale x 2 x float> } @llvm.sincos.nxv2f32(<vscale x 2 x float> zeroinitializer)
; CHECK-NEXT:    ret { <vscale x 2 x float>, <vscale x 2 x float> } [[RET]]
;
  %ret = call { <vscale x 2 x float>, <vscale x 2 x float> } @llvm.sincos.nxv2f32(<vscale x 2 x float> zeroinitializer)
  ret { <vscale x 2 x float>, <vscale x 2 x float> } %ret
}

define { float, float } @sincos_inf() {
; CHECK-LABEL: define { float, float } @sincos_inf() {
; CHECK-NEXT:    [[RET:%.*]] = call { float, float } @llvm.sincos.f32(float 0x7FF0000000000000)
; CHECK-NEXT:    ret { float, float } [[RET]]
;
  %ret = call { float, float } @llvm.sincos.f32(float 0x7FF0000000000000)
  ret { float, float } %ret
}

define { float, float } @sincos_neginf() {
; CHECK-LABEL: define { float, float } @sincos_neginf() {
; CHECK-NEXT:    [[RET:%.*]] = call { float, float } @llvm.sincos.f32(float 0xFFF0000000000000)
; CHECK-NEXT:    ret { float, float } [[RET]]
;
  %ret = call { float, float } @llvm.sincos.f32(float 0xFFF0000000000000)
  ret { float, float } %ret
}

define { float, float } @sincos_qnan() {
; CHECK-LABEL: define { float, float } @sincos_qnan() {
; CHECK-NEXT:    [[RET:%.*]] = call { float, float } @llvm.sincos.f32(float 0x7FF8000000000000)
; CHECK-NEXT:    ret { float, float } [[RET]]
;
  %ret = call { float, float } @llvm.sincos.f32(float 0x7FF8000000000000)
  ret { float, float } %ret
}

define { float, float } @sincos_snan() {
; CHECK-LABEL: define { float, float } @sincos_snan() {
; CHECK-NEXT:    [[RET:%.*]] = call { float, float } @llvm.sincos.f32(float 0x7FF0000020000000)
; CHECK-NEXT:    ret { float, float } [[RET]]
;
  %ret = call { float, float } @llvm.sincos.f32(float bitcast (i32 2139095041 to float))
  ret { float, float } %ret
}
