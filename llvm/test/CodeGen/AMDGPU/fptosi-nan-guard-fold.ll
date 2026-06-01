; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s

; The AMDGPU v_cvt_i32_f32 / v_cvt_u32_f32 instructions already return 0 for
; NaN inputs. The DAG combiner should fold away isnan guards that select 0 for
; NaN, since the hardware already provides the desired behavior.

; Basic: select (fcmp uno x, 0.0), 0, (fptosi x) -> fptosi x
; CHECK-LABEL: nan_guard_fptosi_f32:
; CHECK: v_cvt_i32_f32
; CHECK-NOT: v_cmp_class
; CHECK-NOT: v_cmp_u_f32
; CHECK-NOT: v_cndmask
; CHECK: s_setpc_b64
define i32 @nan_guard_fptosi_f32(float %x) {
  %conv = fptosi float %x to i32
  %isnan = fcmp uno float %x, 0.0
  %sel = select i1 %isnan, i32 0, i32 %conv
  ret i32 %sel
}

; Ordered form: select (fcmp ord x, 0.0), (fptosi x), 0 -> fptosi x
; CHECK-LABEL: ord_guard_fptosi_f32:
; CHECK: v_cvt_i32_f32
; CHECK-NOT: v_cmp_class
; CHECK-NOT: v_cmp_o_f32
; CHECK-NOT: v_cndmask
; CHECK: s_setpc_b64
define i32 @ord_guard_fptosi_f32(float %x) {
  %conv = fptosi float %x to i32
  %isord = fcmp ord float %x, 0.0
  %sel = select i1 %isord, i32 %conv, i32 0
  ret i32 %sel
}

; With AND mask (the actual device-libs pattern):
; select (fcmp uno x, 0.0), 0, (and (fptosi x), 3) -> and (fptosi x), 3
; CHECK-LABEL: nan_guard_fptosi_and_mask:
; CHECK: v_cvt_i32_f32
; CHECK-NOT: v_cmp_class
; CHECK-NOT: v_cmp_u_f32
; CHECK-NOT: v_cndmask
; CHECK: v_and_b32
; CHECK: s_setpc_b64
define i32 @nan_guard_fptosi_and_mask(float %x) {
  %conv = fptosi float %x to i32
  %and = and i32 %conv, 3
  %isnan = fcmp uno float %x, 0.0
  %sel = select i1 %isnan, i32 0, i32 %and
  ret i32 %sel
}

; fptoui variant
; CHECK-LABEL: nan_guard_fptoui_f32:
; CHECK: v_cvt_u32_f32
; CHECK-NOT: v_cmp_class
; CHECK-NOT: v_cmp_u_f32
; CHECK-NOT: v_cndmask
; CHECK: s_setpc_b64
define i32 @nan_guard_fptoui_f32(float %x) {
  %conv = fptoui float %x to i32
  %isnan = fcmp uno float %x, 0.0
  %sel = select i1 %isnan, i32 0, i32 %conv
  ret i32 %sel
}

; f64 source
; CHECK-LABEL: nan_guard_fptosi_f64:
; CHECK: v_cvt_i32_f64
; CHECK-NOT: v_cmp_class
; CHECK-NOT: v_cmp_u_f64
; CHECK-NOT: v_cndmask
; CHECK: s_setpc_b64
define i32 @nan_guard_fptosi_f64(double %x) {
  %conv = fptosi double %x to i32
  %isnan = fcmp uno double %x, 0.0
  %sel = select i1 %isnan, i32 0, i32 %conv
  ret i32 %sel
}

; fcmp uno x, x form (also valid NaN check)
; CHECK-LABEL: nan_guard_fptosi_cmp_self:
; CHECK: v_cvt_i32_f32
; CHECK-NOT: v_cmp_class
; CHECK-NOT: v_cmp_u_f32
; CHECK-NOT: v_cndmask
; CHECK: s_setpc_b64
define i32 @nan_guard_fptosi_cmp_self(float %x) {
  %conv = fptosi float %x to i32
  %isnan = fcmp uno float %x, %x
  %sel = select i1 %isnan, i32 0, i32 %conv
  ret i32 %sel
}

; Negative test: non-zero constant in the NaN arm should NOT fold.
; CHECK-LABEL: nan_guard_nonzero_constant:
; CHECK: v_cvt_i32_f32
; CHECK: v_cmp
; CHECK: v_cndmask
; CHECK: s_setpc_b64
define i32 @nan_guard_nonzero_constant(float %x) {
  %conv = fptosi float %x to i32
  %isnan = fcmp uno float %x, 0.0
  %sel = select i1 %isnan, i32 42, i32 %conv
  ret i32 %sel
}

; Negative test: mismatched operands (fcmp on x, fptosi on y) should NOT fold.
; CHECK-LABEL: nan_guard_mismatched_operands:
; CHECK: v_cvt_i32_f32
; CHECK: v_cmp
; CHECK: v_cndmask
; CHECK: s_setpc_b64
define i32 @nan_guard_mismatched_operands(float %x, float %y) {
  %conv = fptosi float %y to i32
  %isnan = fcmp uno float %x, 0.0
  %sel = select i1 %isnan, i32 0, i32 %conv
  ret i32 %sel
}
