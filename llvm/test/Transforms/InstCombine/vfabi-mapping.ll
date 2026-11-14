; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare double @sin(double) #0
declare double @cos(double) #1

; The VFABI mapping is on the callee declaration, so the call has to stay a
; call, otherwise the loop vectorizer cannot replace it with a vector variant.
define double @mapped_sin(double %x) {
; CHECK-LABEL: @mapped_sin(
; CHECK-NEXT:    [[CALL:%.*]] = call double @sin(double %x)
; CHECK-NEXT:    ret double [[CALL]]
  %r = call double @sin(double %x)
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

attributes #0 = { memory(none) "vector-function-abi-variant"="_ZGV_LLVM_N2v_sin(vector_sin)" }
attributes #1 = { memory(none) }
