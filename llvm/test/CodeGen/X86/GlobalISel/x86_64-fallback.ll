; RUN: llc -O0 -mtriple=x86_64-linux-gnu -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err
; This file checks that the fallback path to selection dag works.
; The test is fragile in the sense that it must be updated to expose
; something that fails with global-isel.
; When we cannot produce a test case anymore, that means we can remove
; the fallback path.

; Check that we fallback on byVal argument
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: call: '  call void @ScaleObjectOverwrite_3(ptr %index, ptr byval(%struct.PointListStruct) %index)' (in function: ScaleObjectOverwrite_2)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for ScaleObjectOverwrite_2
; FALLBACK-WITH-REPORT-OUT-LABEL: ScaleObjectOverwrite_2:
%struct.PointListStruct = type { ptr, ptr }
declare void @ScaleObjectOverwrite_3(ptr %index, ptr byval(%struct.PointListStruct) %index2)
define void @ScaleObjectOverwrite_2(ptr %index) {
entry:
  call void @ScaleObjectOverwrite_3(ptr %index, ptr byval(%struct.PointListStruct) %index)
  ret void
}
