; RUN: llc -march=mips -mcpu=mips32   < %s | FileCheck %s -check-prefixes=ALL,FCC
; RUN: llc -march=mips -mcpu=mips32r2 < %s | FileCheck %s -check-prefixes=ALL,FCC
; RUN: llc -march=mips -mcpu=mips32r6 < %s | FileCheck %s -check-prefixes=ALL,GPR,32-GPR
; RUN: llc -march=mips64 -mcpu=mips4    < %s | FileCheck %s -check-prefixes=ALL,FCC
; RUN: llc -march=mips64 -mcpu=mips64   < %s | FileCheck %s -check-prefixes=ALL,FCC
; RUN: llc -march=mips64 -mcpu=mips64r2 < %s | FileCheck %s -check-prefixes=ALL,FCC
; RUN: llc -march=mips64 -mcpu=mips64r6 < %s | FileCheck %s -check-prefixes=ALL,GPR,64-GPR

define double @foo(double %a, double %b) nounwind readnone {
entry:
; ALL-LABEL: foo:

; FCC:           bc1f {{\$|\.L}}BB
; FCC:           nop

; 32-GPR:        mtc1      $zero, $[[Z:f[0-9]]]
; 32-GPR:        mthc1     $zero, $[[Z:f[0-9]]]
; 64-GPR:        dmtc1     $zero, $[[Z:f[0-9]]]
; GPR:           cmp.lt.d  $[[FGRCC:f[0-9]+]], $[[Z]], $f12
; GPR:           mfc1      $[[GPRCC:[0-9]+]], $[[FGRCC]]
; GPR-NOT:       not       $[[GPRCC]], $[[GPRCC]]
; GPR:           bnezc     $[[GPRCC]], {{\$|\.L}}BB

  %cmp = fcmp ogt double %a, 0.000000e+00
  br i1 %cmp, label %if.end6, label %if.else

if.else:                                          ; preds = %entry
  %cmp3 = fcmp ogt double %b, 0.000000e+00
  br i1 %cmp3, label %if.end6, label %return

if.end6:                                          ; preds = %if.else, %entry
  %c.0 = phi double [ %a, %entry ], [ 0.000000e+00, %if.else ]
  %sub = fsub double %b, %c.0
  %mul = fmul double %sub, 2.000000e+00
  br label %return

return:                                           ; preds = %if.else, %if.end6
  %retval.0 = phi double [ %mul, %if.end6 ], [ 0.000000e+00, %if.else ]
  ret double %retval.0
}

define void @f1(float %f) nounwind {
entry:
; ALL-LABEL: f1:

; FCC:           bc1f {{\$|\.L}}BB
; FCC:           nop

; GPR:           mtc1     $zero, $[[Z:f[0-9]]]
; GPR:           cmp.eq.s $[[FGRCC:f[0-9]+]], $f12, $[[Z]]
; GPR:           mfc1     $[[GPRCC:[0-9]+]], $[[FGRCC]]
; GPR-NOT:       not      $[[GPRCC]], $[[GPRCC]]
; 64-GPR         beqzc    $[[GPRCC]], $BB
; 32-GPR         beqz     $[[GPRCC]], $BB

  %cmp = fcmp une float %f, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @abort() noreturn
  unreachable

if.end:                                           ; preds = %entry
  tail call void (...) @f2() nounwind
  ret void
}

declare void @abort() noreturn nounwind

declare void @f2(...)
