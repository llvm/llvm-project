; RUN: llc < %s -mtriple=thumbv7-none-eabi   -mcpu=cortex-m3 | FileCheck %s -check-prefix=CHECK -check-prefix=NONE -check-prefix=NOREGS
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m4 | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=SP -check-prefix=VFP4-ALL
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m7 | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=DP -check-prefix=FP-ARMv8
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-a8 | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=DP -check-prefix=VFP4-ALL -check-prefix=VFP4-DP
; RUN: llc < %s -mtriple=thumbv8.1m.main-none-eabihf -mattr=+mve | FileCheck %s -check-prefix=CHECK -check-prefix=NONE -check-prefix=ONLYREGS

define float @add_f(float %a, float %b) {
entry:
; CHECK-LABEL: add_f:
; NONE: {{b|bl}} __aeabi_fadd
; HARD: vadd.f32  s0, s0, s1
  %0 = fadd float %a, %b
  ret float %0
}

define double @add_d(double %a, double %b) {
entry:
; CHECK-LABEL: add_d:
; NONE: {{b|bl}} __aeabi_dadd
; SP: {{b|bl}} __aeabi_dadd
; DP: vadd.f64  d0, d0, d1
  %0 = fadd double %a, %b
  ret double %0
}

define float @sub_f(float %a, float %b) {
entry:
; CHECK-LABEL: sub_f:
; NONE: {{b|bl}} __aeabi_fsub
; HARD: vsub.f32  s
  %0 = fsub float %a, %b
  ret float %0
}

define double @sub_d(double %a, double %b) {
entry:
; CHECK-LABEL: sub_d:
; NONE: {{b|bl}} __aeabi_dsub
; SP: {{b|bl}} __aeabi_dsub
; DP: vsub.f64  d0, d0, d1
  %0 = fsub double %a, %b
  ret double %0
}

define float @mul_f(float %a, float %b) {
entry:
; CHECK-LABEL: mul_f:
; NONE: {{b|bl}} __aeabi_fmul
; HARD: vmul.f32  s
  %0 = fmul float %a, %b
  ret float %0
}

define double @mul_d(double %a, double %b) {
entry:
; CHECK-LABEL: mul_d:
; NONE: {{b|bl}} __aeabi_dmul
; SP: {{b|bl}} __aeabi_dmul
; DP: vmul.f64  d0, d0, d1
  %0 = fmul double %a, %b
  ret double %0
}

define float @div_f(float %a, float %b) {
entry:
; CHECK-LABEL: div_f:
; NONE: {{b|bl}} __aeabi_fdiv
; HARD: vdiv.f32  s
  %0 = fdiv float %a, %b
  ret float %0
}

define double @div_d(double %a, double %b) {
entry:
; CHECK-LABEL: div_d:
; NONE: {{b|bl}} __aeabi_ddiv
; SP: {{b|bl}} __aeabi_ddiv
; DP: vdiv.f64  d0, d0, d1
  %0 = fdiv double %a, %b
  ret double %0
}

define float @rem_f(float %a, float %b) {
entry:
; CHECK-LABEL: rem_f:
; NONE: {{b|bl}} fmodf
; HARD: b fmodf
  %0 = frem float %a, %b
  ret float %0
}

define double @rem_d(double %a, double %b) {
entry:
; CHECK-LABEL: rem_d:
; NONE: {{b|bl}} fmod
; HARD: b fmod
  %0 = frem double %a, %b
  ret double %0
}

; In the ONLYREGS case (where we have integer MVE but no floating
; point), we still expect the hard float ABI, because we asked for it
; in the triple, and since the FP registers exist, it's possible to
; use them to pass arguments. So the generated code should load the
; return value into s0, not r0. Similarly for the other load and store
; tests.
define float @load_f(ptr %a) {
entry:
; CHECK-LABEL: load_f:
; NOREGS: ldr r0, [r0]
; ONLYREGS: vldr s0, [r0]
; HARD: vldr s0, [r0]
  %0 = load float, ptr %a, align 4
  ret float %0
}

define double @load_d(ptr %a) {
entry:
; CHECK-LABEL: load_d:
; NOREGS: ldm r0, {r0, r1}
; ONLYREGS: vldr d0, [r0]
; HARD: vldr d0, [r0]
  %0 = load double, ptr %a, align 8
  ret double %0
}

define void @store_f(ptr %a, float %b) {
entry:
; CHECK-LABEL: store_f:
; NOREGS: str r1, [r0]
; ONLYREGS: vstr s0, [r0]
; HARD: vstr s0, [r0]
  store float %b, ptr %a, align 4
  ret void
}

define void @store_d(ptr %a, double %b) {
entry:
; CHECK-LABEL: store_d:
; NOREGS: strd r2, r3, [r0]
; ONLYREGS: vstr d0, [r0]
; HARD: vstr d0, [r0]
  store double %b, ptr %a, align 8
  ret void
}

define double @f_to_d(float %a) {
; CHECK-LABEL: f_to_d:
; NONE: bl __aeabi_f2d
; SP: bl __aeabi_f2d
; DP: vcvt.f64.f32 d0, s0
  %1 = fpext float %a to double
  ret double %1
}

define float @d_to_f(double %a) {
; CHECK-LABEL: d_to_f:
; NONE: bl __aeabi_d2f
; SP: bl __aeabi_d2f
; DP: vcvt.f32.f64 s0, d0
  %1 = fptrunc double %a to float
  ret float %1
}

define i32 @f_to_si(float %a) {
; CHECK-LABEL: f_to_si:
; NONE: bl __aeabi_f2iz
; HARD: vcvt.s32.f32 s0, s0
; HARD: vmov r0, s0
  %1 = fptosi float %a to i32
  ret i32 %1
}

define i32 @d_to_si(double %a) {
; CHECK-LABEL: d_to_si:
; NONE: bl __aeabi_d2iz
; SP: vmov r0, r1, d0
; SP: bl __aeabi_d2iz
; DP: vcvt.s32.f64 s0, d0
; DP: vmov r0, s0
  %1 = fptosi double %a to i32
  ret i32 %1
}

define i32 @f_to_ui(float %a) {
; CHECK-LABEL: f_to_ui:
; NONE: bl __aeabi_f2uiz
; HARD: vcvt.u32.f32 s0, s0
; HARD: vmov r0, s0
  %1 = fptoui float %a to i32
  ret i32 %1
}

define i32 @d_to_ui(double %a) {
; CHECK-LABEL: d_to_ui:
; NONE: bl __aeabi_d2uiz
; SP: vmov r0, r1, d0
; SP: bl __aeabi_d2uiz
; DP: vcvt.u32.f64 s0, d0
; DP: vmov r0, s0
  %1 = fptoui double %a to i32
  ret i32 %1
}

define float @si_to_f(i32 %a) {
; CHECK-LABEL: si_to_f:
; NONE: bl __aeabi_i2f
; HARD: vcvt.f32.s32 s0, s0
  %1 = sitofp i32 %a to float
  ret float %1
}

define double @si_to_d(i32 %a) {
; CHECK-LABEL: si_to_d:
; NONE: bl __aeabi_i2d
; SP: bl __aeabi_i2d
; DP: vcvt.f64.s32 d0, s0
  %1 = sitofp i32 %a to double
  ret double %1
}

define float @ui_to_f(i32 %a) {
; CHECK-LABEL: ui_to_f:
; NONE: bl __aeabi_ui2f
; HARD: vcvt.f32.u32 s0, s0
  %1 = uitofp i32 %a to float
  ret float %1
}

define double @ui_to_d(i32 %a) {
; CHECK-LABEL: ui_to_d:
; NONE: bl __aeabi_ui2d
; SP: bl __aeabi_ui2d
; DP: vcvt.f64.u32 d0, s0
  %1 = uitofp i32 %a to double
  ret double %1
}

define float @bitcast_i_to_f(i32 %a) {
; CHECK-LABEL: bitcast_i_to_f:
; NOREGS-NOT: mov
; ONLYREGS: vmov s0, r0
; HARD: vmov s0, r0
  %1 = bitcast i32 %a to float
  ret float %1
}

define double @bitcast_i_to_d(i64 %a) {
; CHECK-LABEL: bitcast_i_to_d:
; NOREGS-NOT: mov
; ONLYREGS: vmov d0, r0, r1
; HARD: vmov d0, r0, r1
 %1 = bitcast i64 %a to double
  ret double %1
}

define i32 @bitcast_f_to_i(float %a) {
; CHECK-LABEL: bitcast_f_to_i:
; NOREGS-NOT: mov
; ONLYREGS: vmov r0, s0
; HARD: vmov r0, s0
  %1 = bitcast float %a to i32
  ret i32 %1
}

define i64 @bitcast_d_to_i(double %a) {
; CHECK-LABEL: bitcast_d_to_i:
; NOREGS-NOT: mov
; ONLYREGS: vmov r0, r1, d0
; HARD: vmov r0, r1, d0
  %1 = bitcast double %a to i64
  ret i64 %1
}

define float @select_f(float %a, float %b, i1 %c) {
; CHECK-LABEL: select_f:
; NOREGS: lsls    r2, r2, #31
; NOREGS: moveq   r0, r1
; ONLYREGS: lsls    r0, r0, #31
; ONLYREGS: vmovne.f32      s1, s0
; HARD: lsls    r0, r0, #31
; VFP4-ALL: vmovne.f32      s1, s0
; VFP4-ALL: vmov.f32        s0, s1
; FP-ARMv8: vseleq.f32 s0, s1, s0
  %1 = select i1 %c, float %a, float %b
  ret float %1
}

define double @select_d(double %a, double %b, i1 %c) {
; CHECK-LABEL: select_d:
; NOREGS: ldr{{(.w)?}}     [[REG:r[0-9]+]], [sp]
; NOREGS: lsls.w    [[REG]], [[REG]], #31
; ONLYREGS: lsls    r0, r0, #31
; NOREGS-DAG: moveq   r0, r2
; NOREGS-DAG: moveq   r1, r3
; ONLYREGS-DAG: csel   r0, r2, r1
; ONLYREGS-DAG: csel   r1, r12, r3
; SP-DAG: vmov [[ALO:r[0-9]+]], [[AHI:r[0-9]+]], d0
; SP-DAG: vmov [[BLO:r[0-9]+]], [[BHI:r[0-9]+]], d1
; SP: lsls r0, r0, #31
; SP: itt ne
; SP-DAG: movne [[BLO]], [[ALO]]
; SP-DAG: movne [[BHI]], [[AHI]]
; SP: vmov d0, [[BLO]], [[BHI]]
; DP: lsls   r0, r0, #31
; VFP4-DP: vmovne.f64      d1, d0
; VFP4-DP: vmov.f64        d0, d1
; FP-ARMV8: vseleq.f64      d0, d1, d0
  %1 = select i1 %c, double %a, double %b
  ret double %1
}
