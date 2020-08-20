; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define float @selectccaf(float, float, float, float) {
; CHECK-LABEL: selectccaf:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp false float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccat(float, float, float, float) {
; CHECK-LABEL: selectccat:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp true float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccoeq(float, float, float, float) {
; CHECK-LABEL: selectccoeq:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp oeq float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccone(float, float, float, float) {
; CHECK-LABEL: selectccone:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.ne %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp one float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccogt(float, float, float, float) {
; CHECK-LABEL: selectccogt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ogt float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccoge(float, float, float, float) {
; CHECK-LABEL: selectccoge:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.ge %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp oge float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccolt(float, float, float, float) {
; CHECK-LABEL: selectccolt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp olt float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccole(float, float, float, float) {
; CHECK-LABEL: selectccole:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.le %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ole float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccord(float, float, float, float) {
; CHECK-LABEL: selectccord:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s0
; CHECK-NEXT:    cmov.s.num %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ord float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccuno(float, float, float, float) {
; CHECK-LABEL: selectccuno:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s0
; CHECK-NEXT:    cmov.s.nan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp uno float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccueq(float, float, float, float) {
; CHECK-LABEL: selectccueq:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eqnan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ueq float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccune(float, float, float, float) {
; CHECK-LABEL: selectccune:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.nenan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp une float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccugt(float, float, float, float) {
; CHECK-LABEL: selectccugt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.gtnan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ugt float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccuge(float, float, float, float) {
; CHECK-LABEL: selectccuge:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.genan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp uge float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccult(float, float, float, float) {
; CHECK-LABEL: selectccult:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.ltnan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ult float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}

define float @selectccule(float, float, float, float) {
; CHECK-LABEL: selectccule:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.lenan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ule float %0, 0.0
  %6 = select i1 %5, float %2, float %3
  ret float %6
}
