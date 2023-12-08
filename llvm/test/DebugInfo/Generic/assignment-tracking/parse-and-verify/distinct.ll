; RUN: not opt -S %s 2>&1 \
; RUN: | FileCheck %s

;; Check that badly formed assignment tracking metadata is caught either
;; while parsing or by the verifier.

; CHECK: error: missing 'distinct', required for !DIAssignID()

!1 = !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
