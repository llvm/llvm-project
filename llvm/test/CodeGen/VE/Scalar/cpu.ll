; RUN: llc -mtriple=ve -mcpu=help < %s 2>&1 | FileCheck %s

; CHECK: Available CPUs for this target:
; CHECK-EMPTY:
; CHECK-NEXT: generic - Select the generic processor.
