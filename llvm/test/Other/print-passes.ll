; RUN: opt --print-passes | FileCheck %s

; CHECK: Module passes:
; CHECK: no-op-module
; CHECK: Module analyses:
; CHECK: no-op-module
; CHECK: Module alias analyses:
; CHECK: globals-aa
; CHECK: CGSCC passes:
; CHECK: no-op-cgscc
; CHECK: CGSCC analyses:
; CHECK: no-op-cgscc
; CHECK: Function passes:
; CHECK: no-op-function
; CHECK: Function analyses:
; CHECK: no-op-function
; CHECK: Function alias analyses:
; CHECK: basic-aa
; CHECK: Loop passes:
; CHECK: no-op-loop
; CHECK: Loop analyses:
; CHECK: no-op-loop
