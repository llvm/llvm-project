; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mattr=-crbits < %s | FileCheck %s
;
; PR14751: Unsupported type in SelectionDAG::getConstantFP()

define fastcc void @_D3std4math4sqrtFNaNbNfcZc() {
entry:
  br i1 undef, label %if, label %else
; CHECK: cmplwi 3, 0
if:                                               ; preds = %entry
  store { ppc_fp128, ppc_fp128 } zeroinitializer, ptr undef
  ret void

else:                                             ; preds = %entry
  unreachable
}
