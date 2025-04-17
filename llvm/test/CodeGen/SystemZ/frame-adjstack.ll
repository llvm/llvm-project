; RUN: llc < %s -mtriple=s390x-linux-gnu -verify-machineinstrs | FileCheck %s
;
; Test that inserting a new MBB near a call during finalize isel custom
; insertion does not cause all frame instructions to be missed. That would
; result in a missing to set the AdjustsStack flag.

; CHECK-LABEL: fun
define void @fun(i1 %cc) {
  %sel = select i1 %cc, i32 5, i32 0
  tail call void @input_report_abs(i32 %sel)
  %sel2 = select i1 %cc, i32 6, i32 1
  tail call void @input_report_abs(i32 %sel2)
  ret void
}

declare void @input_report_abs(i32)
