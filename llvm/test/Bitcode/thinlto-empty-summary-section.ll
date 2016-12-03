; Ensure we get a summary block even when the file is empty.
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s

; CHECK: <GLOBALVAL_SUMMARY_BLOCK
; CHECK: <VERSION op0=
; CHECK: </GLOBALVAL_SUMMARY_BLOCK>
