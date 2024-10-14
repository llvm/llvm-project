; RUN: opt -passes=sandbox-vectorizer -sbvec-print-pass-pipeline -sbvec-passes="bottom-up-vec<null,null>" %s -disable-output | FileCheck %s

; !!!WARNING!!! This won't get updated by update_test_checks.py !

; This checks the user defined pass pipeline.
define void @pipeline() {
; CHECK: fpm
; CHECK: bottom-up-vec
; CHECK: rpm
; CHECK: null
; CHECK: null
; CHECK-EMPTY:
  ret void
}
