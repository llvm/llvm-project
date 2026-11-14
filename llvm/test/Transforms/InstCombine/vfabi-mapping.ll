; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare double @sin(double) #0
declare double @cos(double) #0
declare double @acosh(double) #0
declare float @acoshf(float)

; A call carrying a VFABI mapping must stay a call, otherwise the loop
; vectorizer cannot replace it with the vector variant.
define double @mapped_sin(double %x) {
; CHECK-LABEL: @mapped_sin(
; CHECK-NEXT:    [[CALL:%.*]] = call double @sin(double %x) #[[MAPPED:[0-9]+]]
; CHECK-NEXT:    ret double [[CALL]]
  %r = call double @sin(double %x) #1
  ret double %r
}

; A call without a mapping is folded to the intrinsic as usual.
define double @plain_cos(double %x) {
; CHECK-LABEL: @plain_cos(
; CHECK:         [[CALL:%.*]] = call double @llvm.cos.f64(double %x)
; CHECK-NEXT:    ret double [[CALL]]
  %r = call double @cos(double %x)
  ret double %r
}

; Rewriting f(-x) to -f(x) must keep the mapping on the new call.
define double @mapped_sin_neg(double %x) {
; CHECK-LABEL: @mapped_sin_neg(
; CHECK-NEXT:    [[CALL:%.*]] = call double @sin(double %x) #[[MAPPED]]
; CHECK-NEXT:    [[NEG:%.*]] = fneg double [[CALL]]
; CHECK-NEXT:    ret double [[NEG]]
  %n = fneg double %x
  %r = call double @sin(double %n) #1
  ret double %r
}

define double @plain_sin_neg(double %x) {
; CHECK-LABEL: @plain_sin_neg(
; CHECK-NEXT:    [[CALL:%.*]] = call double @llvm.sin.f64(double %x)
; CHECK-NEXT:    [[NEG:%.*]] = fneg double [[CALL]]
; CHECK-NEXT:    ret double [[NEG]]
  %n = fneg double %x
  %r = call double @sin(double %n)
  ret double %r
}

; Shrinking a mapped call to its float version would lose the mapping.
define float @mapped_acosh_to_float(float %f) {
; CHECK-LABEL: @mapped_acosh_to_float(
; CHECK-NEXT:    [[CONV:%.*]] = fpext float %f to double
; CHECK-NEXT:    [[CALL:%.*]] = call fast double @acosh(double [[CONV]]) #[[ACOSH_MAPPED:[0-9]+]]
; CHECK-NEXT:    [[TRUNC:%.*]] = fptrunc double [[CALL]] to float
; CHECK-NEXT:    ret float [[TRUNC]]
  %conv = fpext float %f to double
  %r = call fast double @acosh(double %conv) #2
  %t = fptrunc double %r to float
  ret float %t
}

; Without a mapping the shrink still happens.
define float @plain_acosh_to_float(float %f) {
; CHECK-LABEL: @plain_acosh_to_float(
; CHECK-NEXT:    [[SHRUNK:%.*]] = call fast float @acoshf(float %f)
; CHECK-NEXT:    ret float [[SHRUNK]]
  %conv = fpext float %f to double
  %r = call fast double @acosh(double %conv)
  %t = fptrunc double %r to float
  ret float %t
}

; CHECK-DAG: attributes #[[MAPPED]] = { {{.*}}"vector-function-abi-variant"="_ZGV_LLVM_N2v_sin(vector_sin)"
; CHECK-DAG: attributes #[[ACOSH_MAPPED]] = { {{.*}}"vector-function-abi-variant"="_ZGV_LLVM_N4v_acosh(vector_acosh)"

attributes #0 = { memory(none) }
attributes #1 = { "vector-function-abi-variant"="_ZGV_LLVM_N2v_sin(vector_sin)" }
attributes #2 = { "vector-function-abi-variant"="_ZGV_LLVM_N4v_acosh(vector_acosh)" }
