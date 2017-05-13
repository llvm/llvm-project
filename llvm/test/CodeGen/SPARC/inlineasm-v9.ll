; RUN: llc -march=sparcv9 <%s | FileCheck %s

;; Ensures that inline-asm accepts and uses 'f' and 'e' register constraints.
; CHECK-LABEL: faddd:
; CHECK: faddd  %f0, %f2, %f0
define double @faddd(double, double) local_unnamed_addr #2 {
entry:
  %2 = tail call double asm sideeffect "faddd  $1, $2, $0;", "=f,f,e"(double %0, double %1) #7
  ret double %2
}

; CHECK-LABEL: faddq:
; CHECK: faddq  %f0, %f4, %f0
define fp128 @faddq(fp128, fp128) local_unnamed_addr #2 {
entry:
  %2 = tail call fp128 asm sideeffect "faddq  $1, $2, $0;", "=f,f,e"(fp128 %0, fp128 %1) #7
  ret fp128 %2
}

;; Ensure that 'e' can indeed go in the high area, and 'f' cannot.
; CHECK-LABEL: faddd_high:
; CHECK: fmovd  %f2, %f32
; CHECK: fmovd  %f0, %f2
; CHECK: faddd  %f2, %f32, %f2
define double @faddd_high(double, double) local_unnamed_addr #2 {
entry:
  %2 = tail call double asm sideeffect "faddd  $1, $2, $0;", "=f,f,e,~{d0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7}"(double %0, double %1) #7
  ret double %2
}

