; RUN: opt < %s -passes=instcombine -S | FileCheck %s
; This is a check to assure the attributes of `pow` do
; not get passed to sqrt.

define void @pow_to_sqrt(double %x) {
; CHECK-LABEL: @pow_to_sqrt(
; CHECK-NEXT: [[SQRT:%.*]] = call afn double @sqrt(double [[X:%.*]])
; CHECK-NEXT: ret void
;
  %call = call afn double @pow(double %x, double 1.5)
  ret void
}

declare double @pow(double noundef, double noundef)
