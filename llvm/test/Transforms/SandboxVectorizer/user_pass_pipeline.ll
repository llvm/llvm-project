; RUN: opt -passes=sandbox-vectorizer -sbvec-print-pass-pipeline -sbvec-passes=null,null %s -disable-output | FileCheck %s

; !!!WARNING!!! This won't get updated by update_test_checks.py !

; This checks the user defined pass pipeline.
define void @pipeline() {
; CHECK: rpm
; CHECK: null
; CHECK: null
; CHECK-EMPTY:
  ret void
}
