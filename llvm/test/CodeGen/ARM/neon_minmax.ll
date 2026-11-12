; RUN: llc -mtriple=arm-eabi -mcpu=swift %s -o - | FileCheck %s
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 -mattr=-neon %s -o -

define float @fmin_ole(float %x) nounwind {
;CHECK-LABEL: fmin_ole:
;CHECK-NOT: vmin.f32
  %cond = fcmp ole float 1.0, %x
  %min1 = select i1 %cond, float 1.0, float %x
  ret float %min1
}

define float @fmin_ole_zero(float %x) nounwind {
;CHECK-LABEL: fmin_ole_zero:
;CHECK-NOT: vmin.f32
  %cond = fcmp ole float 0.0, %x
  %min1 = select i1 %cond, float 0.0, float %x
  ret float %min1
}

define float @fmin_ult(float %x) nounwind {
;CHECK-LABEL: fmin_ult:
;CHECK-NOT: vmin.f32
  %cond = fcmp ult float %x, 1.0
  %min1 = select i1 %cond, float %x, float 1.0
  ret float %min1
}

define float @fmax_ogt(float %x) nounwind {
;CHECK-LABEL: fmax_ogt:
;CHECK-NOT: vmax.f32
  %cond = fcmp ogt float 1.0, %x
  %max1 = select i1 %cond, float 1.0, float %x
  ret float %max1
}

define float @fmax_uge(float %x) nounwind {
;CHECK-LABEL: fmax_uge:
;CHECK-NOT: vmax.f32
  %cond = fcmp uge float %x, 1.0
  %max1 = select i1 %cond, float %x, float 1.0
  ret float %max1
}

define float @fmax_uge_zero(float %x) nounwind {
;CHECK-LABEL: fmax_uge_zero:
;CHECK-NOT: vmax.f32
  %cond = fcmp uge float %x, 0.0
  %max1 = select i1 %cond, float %x, float 0.0
  ret float %max1
}

define float @fmax_olt_reverse(float %x) nounwind {
;CHECK-LABEL: fmax_olt_reverse:
;CHECK-NOT: vmax.f32
  %cond = fcmp olt float %x, 1.0
  %max1 = select i1 %cond, float 1.0, float %x
  ret float %max1
}

define float @fmax_ule_reverse(float %x) nounwind {
;CHECK-LABEL: fmax_ule_reverse:
;CHECK-NOT: vmax.f32
  %cond = fcmp ult float 1.0, %x
  %max1 = select i1 %cond, float %x, float 1.0
  ret float %max1
}

define float @fmin_oge_reverse(float %x) nounwind {
;CHECK-LABEL: fmin_oge_reverse:
;CHECK-NOT: vmin.f32
  %cond = fcmp oge float %x, 1.0
  %min1 = select i1 %cond, float 1.0, float %x
  ret float %min1
}

define float @fmin_ugt_reverse(float %x) nounwind {
;CHECK-LABEL: fmin_ugt_reverse:
;CHECK-NOT: vmin.f32
  %cond = fcmp ugt float 1.0, %x
  %min1 = select i1 %cond, float %x, float 1.0
  ret float %min1
}
