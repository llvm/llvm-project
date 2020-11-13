; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define double @selectccaf(double, double, double, double) {
; CHECK-LABEL: selectccaf:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp false double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccat(double, double, double, double) {
; CHECK-LABEL: selectccat:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp true double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccoeq(double, double, double, double) {
; CHECK-LABEL: selectccoeq:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp oeq double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccone(double, double, double, double) {
; CHECK-LABEL: selectccone:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.ne %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp one double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccogt(double, double, double, double) {
; CHECK-LABEL: selectccogt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ogt double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccoge(double, double, double, double) {
; CHECK-LABEL: selectccoge:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.ge %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp oge double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccolt(double, double, double, double) {
; CHECK-LABEL: selectccolt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp olt double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccole(double, double, double, double) {
; CHECK-LABEL: selectccole:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.le %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ole double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccord(double, double, double, double) {
; CHECK-LABEL: selectccord:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.num %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ord double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccuno(double, double, double, double) {
; CHECK-LABEL: selectccuno:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.nan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp uno double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccueq(double, double, double, double) {
; CHECK-LABEL: selectccueq:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eqnan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ueq double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccune(double, double, double, double) {
; CHECK-LABEL: selectccune:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.nenan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp une double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccugt(double, double, double, double) {
; CHECK-LABEL: selectccugt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.gtnan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ugt double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccuge(double, double, double, double) {
; CHECK-LABEL: selectccuge:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.genan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp uge double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccult(double, double, double, double) {
; CHECK-LABEL: selectccult:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.ltnan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ult double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccule(double, double, double, double) {
; CHECK-LABEL: selectccule:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.lenan %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ule double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}
