; REQUIRES: amdgpu-registered-target

; AMDGPU inserts the fourth instance of dead-mi-elimination pass after detect-dead-lanes
; This checks that the pipeline stops before that.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -enable-new-pm -stop-before=dead-mi-elimination,4 --print-pipeline-passes -filetype=null %s | FileCheck %s

; There is no way to -start-after an inserted pass right now.
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -O3 -enable-new-pm -start-after=dead-mi-elimination,4 --print-pipeline-passes -filetype=null %s


; CHECK: dead-mi-elimination
; CHECK: dead-mi-elimination
; CHECK: dead-mi-elimination
; CHECK-NOT: dead-mi-elimination
