; RUN: not opt -S %s -experimental-assignment-tracking 2>&1 \
; RUN: | FileCheck %s

;; Check that badly formed assignment tracking metadata is caught either
;; while parsing or by the verifier.

; CHECK: error: expected ')' here

!1 = distinct !DIAssignID(0)
