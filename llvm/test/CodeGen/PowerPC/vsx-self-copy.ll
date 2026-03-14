; RUN: llc -mcpu=pwr7 -mattr=+vsx < %s -verify-machineinstrs | FileCheck %s
; RUN: llc -mcpu=pwr7 -mattr=+vsx -fast-isel -O0 < %s -verify-machineinstrs | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @takFP(double %x, double %y, double %z) #0 {
entry:
  br i1 undef, label %if.then, label %return

if.then:                                          ; preds = %if.then, %entry
  %x.tr16 = phi double [ %call, %if.then ], [ %x, %entry ]
  %call = tail call double @takFP(double undef, double undef, double undef)
  %call4 = tail call double @takFP(double undef, double %x.tr16, double undef)
  %cmp = fcmp olt double undef, %call
  br i1 %cmp, label %if.then, label %return

return:                                           ; preds = %if.then, %entry
  %z.tr.lcssa = phi double [ %z, %entry ], [ %call4, %if.then ]
  ret double %z.tr.lcssa

; CHECK: @takFP
; CHECK-NOT: xxlor 0, 0, 0
; CHECK: blr
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

