; RUN: opt -passes=sandbox-vectorizer -sbvec-print-pass-pipeline -sbvec-passes=bottom-up-vec,bottom-up-vec %s -disable-output | FileCheck %s

; !!!WARNING!!! This won't get updated by update_test_checks.py !

; This checks the user defined pass pipeline.
define void @pipeline() {
; CHECK: pm
; CHECK: bottom-up-vec
; CHECK: bottom-up-vec
; CHECK-EMPTY:
  ret void
}
