; RUN: opt -verify -S < %s 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: sed -e s/.T2:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK2 %s
; RUN: sed -e s/.T3:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK3 %s
; RUN: sed -e s/.T4:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK4 %s
; RUN: sed -e s/.T5:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK5 %s

; Common declarations used for all runs.
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)

; Test that the verifier accepts legal code, and that the correct attributes are
; attached to the FP intrinsic.
; CHECK1: declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata) #[[ATTR:[0-9]+]]
; CHECK1: declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata) #[[ATTR]]
; CHECK1: attributes #[[ATTR]] = { inaccessiblememonly nounwind }
; Note: FP exceptions aren't usually caught through normal unwind mechanisms,
;       but we may want to revisit this for asynchronous exception handling.
define double @f1(double %a, double %b) {
entry:
  %fadd = call double @llvm.experimental.constrained.fadd.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %fadd
}

define double @f1u(double %a) {
entry:
  %fsqrt = call double @llvm.experimental.constrained.sqrt.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %fsqrt
}

; Test an illegal value for the rounding mode argument.
; CHECK2: invalid rounding mode argument
;T2: define double @f2(double %a, double %b) {
;T2: entry:
;T2:   %fadd = call double @llvm.experimental.constrained.fadd.f64(
;T2:                                           double %a, double %b,
;T2:                                           metadata !"round.dynomite",
;T2:                                           metadata !"fpexcept.strict")
;T2:   ret double %fadd
;T2: }

; Test an illegal value for the exception behavior argument.
; CHECK3: invalid exception behavior argument
;T3: define double @f3(double %a, double %b) {
;T3: entry:
;T3:   %fadd = call double @llvm.experimental.constrained.fadd.f64(
;T3:                                         double %a, double %b,
;T3:                                         metadata !"round.dynamic",
;T3:                                         metadata !"fpexcept.restrict")
;T3:   ret double %fadd
;T3: }

; Test an illegal value for the rounding mode argument.
; CHECK4: invalid rounding mode argument
;T4: define double @f4(double %a) {
;T4: entry:
;T4:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(
;T4:                                           double %a,
;T4:                                           metadata !"round.dynomite",
;T4:                                           metadata !"fpexcept.strict")
;T4:   ret double %fadd
;T4: }

; Test an illegal value for the exception behavior argument.
; CHECK5: invalid exception behavior argument
;T5: define double @f5(double %a) {
;T5: entry:
;T5:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(
;T5:                                         double %a,
;T5:                                         metadata !"round.dynamic",
;T5:                                         metadata !"fpexcept.restrict")
;T5:   ret double %fadd
;T5: }
