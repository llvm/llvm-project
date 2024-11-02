; RUN: llc < %s -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -verify-machineinstrs | FileCheck %s

@gd = external local_unnamed_addr global [500 x double], align 8
@gf = external local_unnamed_addr global [500 x float], align 4

; Function Attrs: nounwind
define double @_Z7getLXSDddddddddddddd(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, double %j, double %k, double %l, double %m) local_unnamed_addr #0 {
entry:
  %0 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 10), align 8
  %1 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 17), align 8
  %2 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 87), align 8
  %3 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 97), align 8
  %4 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 77), align 8
  store double %3, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 122), align 8
  %add = fadd double %a, %b
  %add1 = fadd double %add, %c
  %add2 = fadd double %add1, %d
  %add3 = fadd double %add2, %e
  %add4 = fadd double %add3, %f
  %add5 = fadd double %add4, %g
  %add6 = fadd double %add5, %h
  %add7 = fadd double %add6, %i
  %add8 = fadd double %add7, %j
  %add9 = fadd double %add8, %k
  %add10 = fadd double %add9, %l
  %add11 = fadd double %add10, %m
  %add12 = fadd double %add11, %0
  %add13 = fadd double %add12, %1
  %add14 = fadd double %add13, %2
  %add15 = fadd double %add14, %3
  %add16 = fadd double %add15, %4
  %call = tail call double @_Z7getLXSDddddddddddddd(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, double %j, double %k, double %l, double %m)
  %add17 = fadd double %add16, %call
  ret double %add17
; CHECK-LABEL: _Z7getLXSDddddddddddddd
; CHECK: lxsd [[LD:[0-9]+]], 776(3)
; CHECK: stxsd [[LD]], 976(3)
}

; Function Attrs: nounwind
define float @_Z8getLXSSPfffffffffffff(float %a, float %b, float %c, float %d, float %e, float %f, float %g, float %h, float %i, float %j, float %k, float %l, float %m) local_unnamed_addr #0 {
entry:
  %0 = load float, ptr getelementptr inbounds ([500 x float], ptr @gf, i64 0, i64 10), align 4
  %1 = load float, ptr getelementptr inbounds ([500 x float], ptr @gf, i64 0, i64 17), align 4
  %2 = load float, ptr getelementptr inbounds ([500 x float], ptr @gf, i64 0, i64 87), align 4
  %3 = load float, ptr getelementptr inbounds ([500 x float], ptr @gf, i64 0, i64 97), align 4
  %4 = load float, ptr getelementptr inbounds ([500 x float], ptr @gf, i64 0, i64 77), align 4
  store float %3, ptr getelementptr inbounds ([500 x float], ptr @gf, i64 0, i64 122), align 4
  %add = fadd float %a, %b
  %add1 = fadd float %add, %c
  %add2 = fadd float %add1, %d
  %add3 = fadd float %add2, %e
  %add4 = fadd float %add3, %f
  %add5 = fadd float %add4, %g
  %add6 = fadd float %add5, %h
  %add7 = fadd float %add6, %i
  %add8 = fadd float %add7, %j
  %add9 = fadd float %add8, %k
  %add10 = fadd float %add9, %l
  %add11 = fadd float %add10, %m
  %add12 = fadd float %add11, %0
  %add13 = fadd float %add12, %1
  %add14 = fadd float %add13, %2
  %add15 = fadd float %add14, %3
  %add16 = fadd float %add15, %4
  %call = tail call float @_Z8getLXSSPfffffffffffff(float %a, float %b, float %c, float %d, float %e, float %f, float %g, float %h, float %i, float %j, float %k, float %l, float %m)
  %add17 = fadd float %add16, %call
  ret float %add17
; CHECK-LABEL: _Z8getLXSSPfffffffffffff
; CHECK: lxssp [[LD:[0-9]+]], 388(3)
; CHECK: stxssp [[LD]], 488(3)
}
