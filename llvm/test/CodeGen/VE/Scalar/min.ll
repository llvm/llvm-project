; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define double @minf64(double, double) {
; CHECK-LABEL: minf64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fmin.d %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp olt double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

define double @min2f64(double, double) {
; CHECK-LABEL: min2f64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fmin.d %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ole double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

define double @minuf64(double, double) {
; CHECK-LABEL: minuf64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s2, %s0, %s1
; CHECK-NEXT:    cmov.d.ltnan %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ult double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

define double @min2uf64(double, double) {
; CHECK-LABEL: min2uf64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s2, %s0, %s1
; CHECK-NEXT:    cmov.d.lenan %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ule double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

define float @minf32(float, float) {
; CHECK-LABEL: minf32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fmin.s %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp olt float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

define float @min2f32(float, float) {
; CHECK-LABEL: min2f32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fmin.s %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ole float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

define float @minuf32(float, float) {
; CHECK-LABEL: minuf32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s2, %s0, %s1
; CHECK-NEXT:    cmov.s.ltnan %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ult float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

define float @min2uf32(float, float) {
; CHECK-LABEL: min2uf32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s2, %s0, %s1
; CHECK-NEXT:    cmov.s.lenan %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ule float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

define i64 @mini64(i64, i64) {
; CHECK-LABEL: mini64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    mins.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp slt i64 %0, %1
  %4 = select i1 %3, i64 %0, i64 %1
  ret i64 %4
}

define i64 @min2i64(i64, i64) {
; CHECK-LABEL: min2i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    mins.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp sle i64 %0, %1
  %4 = select i1 %3, i64 %0, i64 %1
  ret i64 %4
}

define i64 @minu64(i64, i64) {
; CHECK-LABEL: minu64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmpu.l %s2, %s0, %s1
; CHECK-NEXT:    cmov.l.lt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp ult i64 %0, %1
  %4 = select i1 %3, i64 %0, i64 %1
  ret i64 %4
}

define i64 @min2u64(i64, i64) {
; CHECK-LABEL: min2u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmpu.l %s2, %s0, %s1
; CHECK-NEXT:    cmov.l.le %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp ule i64 %0, %1
  %4 = select i1 %3, i64 %0, i64 %1
  ret i64 %4
}

define i32 @mini32(i32, i32) {
; CHECK-LABEL: mini32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    mins.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp slt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define i32 @min2i32(i32, i32) {
; CHECK-LABEL: min2i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    mins.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp sle i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define i32 @minu32(i32, i32) {
; CHECK-LABEL: minu32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmpu.w %s2, %s0, %s1
; CHECK-NEXT:    cmov.w.lt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp ult i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define i32 @min2u32(i32, i32) {
; CHECK-LABEL: min2u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmpu.w %s2, %s0, %s1
; CHECK-NEXT:    cmov.w.le %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp ule i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define zeroext i1 @mini1(i1 zeroext, i1 zeroext) {
; CHECK-LABEL: mini1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s2, %s1, %s0
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = xor i1 %0, true
  %4 = and i1 %3, %1
  %5 = select i1 %4, i1 %0, i1 %1
  ret i1 %5
}
