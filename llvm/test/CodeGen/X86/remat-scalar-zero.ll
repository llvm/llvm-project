; XFAIL: *
; ...should pass. See PR12324: misched bringup
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu > %t
; RUN: not grep xor %t
; RUN: not grep movap %t
; RUN: grep "\.quad.*0" %t

; Remat should be able to fold the zero constant into the div instructions
; as a constant-pool load.

define void @foo(ptr nocapture %x, ptr nocapture %y) nounwind {
entry:
  %tmp1 = load double, ptr %x                         ; <double> [#uses=1]
  %arrayidx4 = getelementptr inbounds double, ptr %x, i64 1 ; <ptr> [#uses=1]
  %tmp5 = load double, ptr %arrayidx4                 ; <double> [#uses=1]
  %arrayidx8 = getelementptr inbounds double, ptr %x, i64 2 ; <ptr> [#uses=1]
  %tmp9 = load double, ptr %arrayidx8                 ; <double> [#uses=1]
  %arrayidx12 = getelementptr inbounds double, ptr %x, i64 3 ; <ptr> [#uses=1]
  %tmp13 = load double, ptr %arrayidx12               ; <double> [#uses=1]
  %arrayidx16 = getelementptr inbounds double, ptr %x, i64 4 ; <ptr> [#uses=1]
  %tmp17 = load double, ptr %arrayidx16               ; <double> [#uses=1]
  %arrayidx20 = getelementptr inbounds double, ptr %x, i64 5 ; <ptr> [#uses=1]
  %tmp21 = load double, ptr %arrayidx20               ; <double> [#uses=1]
  %arrayidx24 = getelementptr inbounds double, ptr %x, i64 6 ; <ptr> [#uses=1]
  %tmp25 = load double, ptr %arrayidx24               ; <double> [#uses=1]
  %arrayidx28 = getelementptr inbounds double, ptr %x, i64 7 ; <ptr> [#uses=1]
  %tmp29 = load double, ptr %arrayidx28               ; <double> [#uses=1]
  %arrayidx32 = getelementptr inbounds double, ptr %x, i64 8 ; <ptr> [#uses=1]
  %tmp33 = load double, ptr %arrayidx32               ; <double> [#uses=1]
  %arrayidx36 = getelementptr inbounds double, ptr %x, i64 9 ; <ptr> [#uses=1]
  %tmp37 = load double, ptr %arrayidx36               ; <double> [#uses=1]
  %arrayidx40 = getelementptr inbounds double, ptr %x, i64 10 ; <ptr> [#uses=1]
  %tmp41 = load double, ptr %arrayidx40               ; <double> [#uses=1]
  %arrayidx44 = getelementptr inbounds double, ptr %x, i64 11 ; <ptr> [#uses=1]
  %tmp45 = load double, ptr %arrayidx44               ; <double> [#uses=1]
  %arrayidx48 = getelementptr inbounds double, ptr %x, i64 12 ; <ptr> [#uses=1]
  %tmp49 = load double, ptr %arrayidx48               ; <double> [#uses=1]
  %arrayidx52 = getelementptr inbounds double, ptr %x, i64 13 ; <ptr> [#uses=1]
  %tmp53 = load double, ptr %arrayidx52               ; <double> [#uses=1]
  %arrayidx56 = getelementptr inbounds double, ptr %x, i64 14 ; <ptr> [#uses=1]
  %tmp57 = load double, ptr %arrayidx56               ; <double> [#uses=1]
  %arrayidx60 = getelementptr inbounds double, ptr %x, i64 15 ; <ptr> [#uses=1]
  %tmp61 = load double, ptr %arrayidx60               ; <double> [#uses=1]
  %arrayidx64 = getelementptr inbounds double, ptr %x, i64 16 ; <ptr> [#uses=1]
  %tmp65 = load double, ptr %arrayidx64               ; <double> [#uses=1]
  %div = fdiv double %tmp1, 0.000000e+00          ; <double> [#uses=1]
  store double %div, ptr %y
  %div70 = fdiv double %tmp5, 2.000000e-01        ; <double> [#uses=1]
  %arrayidx72 = getelementptr inbounds double, ptr %y, i64 1 ; <ptr> [#uses=1]
  store double %div70, ptr %arrayidx72
  %div74 = fdiv double %tmp9, 2.000000e-01        ; <double> [#uses=1]
  %arrayidx76 = getelementptr inbounds double, ptr %y, i64 2 ; <ptr> [#uses=1]
  store double %div74, ptr %arrayidx76
  %div78 = fdiv double %tmp13, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx80 = getelementptr inbounds double, ptr %y, i64 3 ; <ptr> [#uses=1]
  store double %div78, ptr %arrayidx80
  %div82 = fdiv double %tmp17, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx84 = getelementptr inbounds double, ptr %y, i64 4 ; <ptr> [#uses=1]
  store double %div82, ptr %arrayidx84
  %div86 = fdiv double %tmp21, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx88 = getelementptr inbounds double, ptr %y, i64 5 ; <ptr> [#uses=1]
  store double %div86, ptr %arrayidx88
  %div90 = fdiv double %tmp25, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx92 = getelementptr inbounds double, ptr %y, i64 6 ; <ptr> [#uses=1]
  store double %div90, ptr %arrayidx92
  %div94 = fdiv double %tmp29, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx96 = getelementptr inbounds double, ptr %y, i64 7 ; <ptr> [#uses=1]
  store double %div94, ptr %arrayidx96
  %div98 = fdiv double %tmp33, 2.000000e-01       ; <double> [#uses=1]
  %arrayidx100 = getelementptr inbounds double, ptr %y, i64 8 ; <ptr> [#uses=1]
  store double %div98, ptr %arrayidx100
  %div102 = fdiv double %tmp37, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx104 = getelementptr inbounds double, ptr %y, i64 9 ; <ptr> [#uses=1]
  store double %div102, ptr %arrayidx104
  %div106 = fdiv double %tmp41, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx108 = getelementptr inbounds double, ptr %y, i64 10 ; <ptr> [#uses=1]
  store double %div106, ptr %arrayidx108
  %div110 = fdiv double %tmp45, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx112 = getelementptr inbounds double, ptr %y, i64 11 ; <ptr> [#uses=1]
  store double %div110, ptr %arrayidx112
  %div114 = fdiv double %tmp49, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx116 = getelementptr inbounds double, ptr %y, i64 12 ; <ptr> [#uses=1]
  store double %div114, ptr %arrayidx116
  %div118 = fdiv double %tmp53, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx120 = getelementptr inbounds double, ptr %y, i64 13 ; <ptr> [#uses=1]
  store double %div118, ptr %arrayidx120
  %div122 = fdiv double %tmp57, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx124 = getelementptr inbounds double, ptr %y, i64 14 ; <ptr> [#uses=1]
  store double %div122, ptr %arrayidx124
  %div126 = fdiv double %tmp61, 2.000000e-01      ; <double> [#uses=1]
  %arrayidx128 = getelementptr inbounds double, ptr %y, i64 15 ; <ptr> [#uses=1]
  store double %div126, ptr %arrayidx128
  %div130 = fdiv double %tmp65, 0.000000e+00      ; <double> [#uses=1]
  %arrayidx132 = getelementptr inbounds double, ptr %y, i64 16 ; <ptr> [#uses=1]
  store double %div130, ptr %arrayidx132
  ret void
}
