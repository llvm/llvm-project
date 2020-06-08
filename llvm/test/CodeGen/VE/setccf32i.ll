; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define zeroext i1 @setccaf(float, float) {
; CHECK-LABEL: setccaf:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp false float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccat(float, float) {
; CHECK-LABEL: setccat:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp true float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccoeq(float, float) {
; CHECK-LABEL: setccoeq:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.eq %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp oeq float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccone(float, float) {
; CHECK-LABEL: setccone:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.ne %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp one float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccogt(float, float) {
; CHECK-LABEL: setccogt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.gt %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ogt float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccoge(float, float) {
; CHECK-LABEL: setccoge:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.ge %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp oge float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccolt(float, float) {
; CHECK-LABEL: setccolt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.lt %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp olt float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccole(float, float) {
; CHECK-LABEL: setccole:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.le %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ole float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccord(float, float) {
; CHECK-LABEL: setccord:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s1, %s0, %s0
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.num %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ord float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccuno(float, float) {
; CHECK-LABEL: setccuno:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s1, %s0, %s0
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.nan %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp uno float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccueq(float, float) {
; CHECK-LABEL: setccueq:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.eqnan %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ueq float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccune(float, float) {
; CHECK-LABEL: setccune:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.nenan %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp une float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccugt(float, float) {
; CHECK-LABEL: setccugt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.gtnan %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ugt float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccuge(float, float) {
; CHECK-LABEL: setccuge:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.genan %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp uge float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccult(float, float) {
; CHECK-LABEL: setccult:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.ltnan %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ult float %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccule(float, float) {
; CHECK-LABEL: setccule:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.s %s1, %s0, %s1
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.s.lenan %s0, (63)0, %s1
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp ule float %0, 0.0
  ret i1 %3
}
