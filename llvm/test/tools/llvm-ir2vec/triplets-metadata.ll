; RUN: llvm-ir2vec triplets %s | FileCheck %s

; Metadata operands emit no triplet, but the remaining operands keep their
; positions (the callee stays Arg4).

define double @f(double %a, double %b) strictfp {
  %r = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %r
}

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)

; CHECK:      MAX_RELATION=6
; CHECK-NEXT: 57	71	0
; CHECK-NEXT: 57	87	2
; CHECK-NEXT: 57	87	3
; CHECK-NEXT: 57	84	6
; CHECK-NEXT: 57	0	1
; CHECK-NEXT: 0	72	0
; CHECK-NEXT: 0	87	2
