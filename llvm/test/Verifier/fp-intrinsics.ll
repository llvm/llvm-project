; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)

; Test an illegal value for the rounding mode argument.
; CHECK: invalid rounding mode argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynomic", metadata !"fpexcept.strict") #1
define double @f2(double %a, double %b) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.fadd.f64(
                                          double %a, double %b,
                                          metadata !"round.dynomic",
                                          metadata !"fpexcept.strict") #0
  ret double %fadd
}

; Test an illegal value for the exception behavior argument.
; CHECK-NEXT: invalid exception behavior argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynamic", metadata !"fpexcept.restrict") #1
define double @f3(double %a, double %b) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.fadd.f64(
                                        double %a, double %b,
                                        metadata !"round.dynamic",
                                        metadata !"fpexcept.restrict") #0
  ret double %fadd
}

; Test an illegal value for the rounding mode argument.
; CHECK-NEXT: invalid rounding mode argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(double %a, metadata !"round.dynomic", metadata !"fpexcept.strict") #1
define double @f4(double %a) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                          double %a,
                                          metadata !"round.dynomic",
                                          metadata !"fpexcept.strict") #0
  ret double %fadd
}

; Test an illegal value for the exception behavior argument.
; CHECK-NEXT: invalid exception behavior argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(double %a, metadata !"round.dynamic", metadata !"fpexcept.restrict") #1
define double @f5(double %a) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                        double %a,
                                        metadata !"round.dynamic",
                                        metadata !"fpexcept.restrict") #0
  ret double %fadd
}

attributes #0 = { strictfp }
