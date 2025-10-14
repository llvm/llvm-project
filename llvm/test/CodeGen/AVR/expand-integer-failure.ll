; RUN: llc < %s -mtriple=avr | FileCheck %s

; Causes an assertion error
; Assertion failed: (Lo.getValueType() == TLI.getTypeToTransformTo(*DAG.getContext(), Op.getValueType()) &&
;   Hi.getValueType() == Lo.getValueType() &&
;   "Invalid type for expanded integer"),
; function SetExpandedInteger
; file lib/CodeGen/SelectionDAG/LegalizeTypes.cpp

; CHECK-LABEL: foo
define void @foo(i16 %a) {
ifcont:
  %cmp_result = icmp eq i16 %a, 255
  %bool_result = uitofp i1 %cmp_result to double
  %result = fcmp one double 0.000000e+00, %bool_result
  br i1 %result, label %then, label %else
then:
  ret void
else:
  ret void
}
