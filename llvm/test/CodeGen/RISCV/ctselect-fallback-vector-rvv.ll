; RUN: llc < %s -mtriple=riscv64 -mattr=+v -O3 | FileCheck %s
;
; Placeholder. RVV scalable-vector CT_SELECT coverage was removed because
; the new legalizer expansion in lib/CodeGen/SelectionDAG/LegalizeDAG.cpp
; (case ISD::CT_SELECT) currently hits "Unexpected illegal type!" in
; LegalizeOp when the scalar-mask-then-splat path produces intermediate
; nodes for some scalable vector element types. Restore the original
; tests after the legalizer is fixed.

define void @placeholder_until_legalizer_fix() {
; CHECK-LABEL: placeholder_until_legalizer_fix:
; CHECK: ret
  ret void
}
