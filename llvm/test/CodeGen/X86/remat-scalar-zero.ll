; XFAIL: *
; ...should pass. See PR12324: misched bringup
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu > %t
; RUN: not grep xor %t
; RUN: not grep movap %t
; RUN: grep "\.quad.*0" %t

; Remat should be able to fold the zero constant into the div instructions
; as a constant-pool load.

define void @foo(double* nocapture %x, double* nocapture %y) nounwind {
entry:
  %tmp1 = load double* %x                         ; <double> [#uses=1]
  %arrayidx4 = getelementptr inbounds double* %x, i64 1 ; <double*> [#uses=1]
  %tmp5 = load double* %arrayidx4                 ; <double> [#uses=1]
  %arrayidx8 = getelementptr inbounds double* %x, i64 2 ; <double*> [#uses=1]
  %tmp9 = load double* %arrayidx8                 ; <double> [#uses=1]
  %arrayidx12 = getelementptr inbounds double* %x, i64 3 ; <double*> [#uses=1]
  %tmp13 = load double* %arrayidx12               ; <double> [#uses=1]
  %arrayidx16 = getelementptr inbounds double* %x, i64 4 ; <double*> [#uses=1]
  %tmp17 = load double* %arrayidx16               ; <double> [#uses=1]
  %arrayidx20 = getelementptr inbounds double* %x, i64 5 ; <double*> [#uses=1]
  %tmp21 = load double* %arrayidx20               ; <double> [#uses=1]
  %arrayidx24 = getelementptr inbounds double* %x, i64 6 ; <double*> [#uses=1]
  %tmp25 = load double* %arrayidx24               ; <double> [#uses=1]
  %arrayidx28 = getelementptr inbounds double* %x, i64 7 ; <double*> [#uses=1]
  %tmp29 = load double* %arrayidx28               ; <double> [#uses=1]
  %arrayidx32 = getelementptr inbounds double* %x, i64 8 ; <double*> [#uses=1]
  %tmp33 = load double* %arrayidx32               ; <double> [#uses=1]
  %arrayidx36 = getelementptr inbounds double* %x, i64 9 ; <double*> [#uses=1]
  %tmp37 = load double* %arrayidx36               ; <double> [#uses=1]
  %arrayidx40 = getelementptr inbounds double* %x, i64 10 ; <double*> [#uses=1]
  %tmp41 = load double* %arrayidx40               ; <double> [#uses=1]
  %arrayidx44 = getelementptr inbounds double* %x, i64 11 ; <double*> [#uses=1]
  %tmp45 = load double* %arrayidx44               ; <double> [#uses=1]
  %arrayidx48 = getelementptr inbounds double* %x, i64 12 ; <double*> [#uses=1]
  %tmp49 = load double* %arrayidx48               ; <double> [#uses=1]
  %arrayidx52 = getelementptr inbounds double* %x, i64 13 ; <double*> [#uses=1]
  %tmp53 = load double* %arrayidx52               ; <double> [#uses=1]
  %arrayidx56 = getelementptr inbounds double* %x, i64 14 ; <double*> [#uses=1]
  %tmp57 = load double* %arrayidx56               ; <double> [#uses=1]
  %arrayidx60 = getelementptr inbounds double* %x, i64 15 ; <double*> [#uses=1]
  %tmp61 = load double* %arrayidx60               ; <double> [#uses=1]
  %arrayidx64 = getelementptr inbounds double* %x, i64 16 ; <double*> [#uses=1]
  %tmp65 = load double* %arrayidx64               ; <double> [#uses=1]
  %div = fdiv double %tmp1, 0.000000e+00          ; <double> [#uses=1]
  store double %div, double* %y
  %div70 = fdiv double %tmp5, 2.000000e-01        ; <double> [#uses=1]
  %arrayidx72 = getelementptr inbounds double* %y, i64 1 ; <double*> [#uses=1]
  store double %div70, double* %arrayidx72
  %div74 = fdiv double %tmp9, 2.000000e-01        ; <double> [#uses=1]
  %arrayidx76 = getelementptr inbounds double* %y, i64 2 ; <double*> [#uses=1]
  store double %div74, double* %arrayidx76
  %div78 = fdiv double %tmp13, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx80 = getelementptr inbounds double* %y, i64 3 ; <double*> [#uses=1]
  store double %div78, double* %arrayidx80
  %div82 = fdiv double %tmp17, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx84 = getelementptr inbounds double* %y, i64 4 ; <double*> [#uses=1]
  store double %div82, double* %arrayidx84
  %div86 = fdiv double %tmp21, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx88 = getelementptr inbounds double* %y, i64 5 ; <double*> [#uses=1]
  store double %div86, double* %arrayidx88
  %div90 = fdiv double %tmp25, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx92 = getelementptr inbounds double* %y, i64 6 ; <double*> [#uses=1]
  store double %div90, double* %arrayidx92
  %div94 = fdiv double %tmp29, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx96 = getelementptr inbounds double* %y, i64 7 ; <double*> [#uses=1]
  store double %div94, double* %arrayidx96
  %div98 = fdiv double %tmp33, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx100 = getelementptr inbounds double* %y, i64 8 ; <double*> [#uses=1]
  store double %div98, double* %arrayidx100
  %div102 = fdiv double %tmp37, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx104 = getelementptr inbounds double* %y, i64 9 ; <double*> [#uses=1]
  store double %div102, double* %arrayidx104
  %div106 = fdiv double %tmp41, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx108 = getelementptr inbounds double* %y, i64 10 ; <double*> [#uses=1]
  store double %div106, double* %arrayidx108
  %div110 = fdiv double %tmp45, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx112 = getelementptr inbounds double* %y, i64 11 ; <double*> [#uses=1]
  store double %div110, double* %arrayidx112
  %div114 = fdiv double %tmp49, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx116 = getelementptr inbounds double* %y, i64 12 ; <double*> [#uses=1]
  store double %div114, double* %arrayidx116
  %div118 = fdiv double %tmp53, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx120 = getelementptr inbounds double* %y, i64 13 ; <double*> [#uses=1]
  store double %div118, double* %arrayidx120
  %div122 = fdiv double %tmp57, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx124 = getelementptr inbounds double* %y, i64 14 ; <double*> [#uses=1]
  store double %div122, double* %arrayidx124
  %div126 = fdiv double %tmp61, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx128 = getelementptr inbounds double* %y, i64 15 ; <double*> [#uses=1]
  store double %div126, double* %arrayidx128
  %div130 = fdiv double %tmp65, 0.000000e+00      ; <double> [#uses=1]
  %arrayidx132 = getelementptr inbounds double* %y, i64 16 ; <double*> [#uses=1]
  store double %div130, double* %arrayidx132
  ret void
}
