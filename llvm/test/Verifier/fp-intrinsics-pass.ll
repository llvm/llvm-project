; RUN: opt -passes=verify -S < %s 2>&1 | FileCheck %s

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata) #0
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata) #0

; Test that the verifier accepts legal code, and that the correct attributes are
; attached to the FP intrinsic. The attributes are checked at the bottom.
; CHECK: declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata) #[[ATTR:[0-9]+]]
; CHECK: declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata) #[[ATTR]]
; Note: FP exceptions aren't usually caught through normal unwind mechanisms,
;       but we may want to revisit this for asynchronous exception handling.
define double @f1(double %a, double %b) #0 {
; CHECK-LABEL: define double @f1
; CHECK-SAME: (double [[A:%.*]], double [[B:%.*]]) #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FADD:%.*]] = call double @llvm.experimental.constrained.fadd.f64(double [[A]], double [[B]], metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR1]]
; CHECK-NEXT:    ret double [[FADD]]
entry:
  %fadd = call double @llvm.experimental.constrained.fadd.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %fadd
}

define double @f1u(double %a) #0 {
; CHECK-LABEL: define double @f1u
; CHECK-SAME: (double [[A:%.*]]) #[[ATTR1]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FSQRT:%.*]] = call double @llvm.experimental.constrained.sqrt.f64(double [[A]], metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR1]]
; CHECK-NEXT:    ret double [[FSQRT]]
;
entry:
  %fsqrt = call double @llvm.experimental.constrained.sqrt.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %fsqrt
}

attributes #0 = { strictfp }
; TODO: Why is strictfp not in the below list?
; CHECK: attributes #[[ATTR]] = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
