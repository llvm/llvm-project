; RUN: llc < %s -mtriple=arm-apple-ios -arm-atomic-cfg-tidy=0 -mcpu=cortex-a9 | FileCheck %s
; rdar://8402126
; Make sure if-converter is not predicating vldmia and ldmia. These are
; micro-coded and would have long issue latency even if predicated on
; false predicate.

define void @t(double %a, double %b, double %c, double %d, i32* nocapture %solutions, double* nocapture %x) nounwind "no-frame-pointer-elim"="true" {
entry:
; CHECK-LABEL: t:
; CHECK: vpop {d8}
; CHECK-NOT: vpopne
; CHECK: pop {r7, pc}
  br i1 undef, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %mul73 = fmul double undef, 0.000000e+00
  %sub76 = fsub double %mul73, undef
  store double %sub76, double* undef, align 4
  %call88 = tail call double @cos(double 0.000000e+00) nounwind
  %mul89 = fmul double undef, %call88
  %sub92 = fsub double %mul89, undef
  store double %sub92, double* undef, align 4
  ret void

if.else:                                          ; preds = %entry
  %tmp101 = tail call double @llvm.pow.f64(double undef, double 0x3FD5555555555555)
  %add112 = fadd double %tmp101, undef
  %mul118 = fmul double %add112, undef
  store double 0.000000e+00, double* %x, align 4
  ret void
}

declare double @acos(double)

declare double @sqrt(double) readnone

declare double @cos(double) readnone

declare double @fabs(double)

declare double @llvm.pow.f64(double, double) nounwind readonly
