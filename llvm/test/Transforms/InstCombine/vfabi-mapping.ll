; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare double @sin(double) #0
declare double @cos(double) #0

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

attributes #0 = { memory(none) }
attributes #1 = { "vector-function-abi-variant"="_ZGV_LLVM_N2v_sin(vector_sin)" }
