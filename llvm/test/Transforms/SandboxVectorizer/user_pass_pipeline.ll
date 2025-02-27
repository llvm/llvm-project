; RUN: opt -passes=sandbox-vectorizer -sbvec-print-pass-pipeline \
; RUN:     -disable-output -sbvec-passes="seed-collection<null,null>" %s \
; RUN:     | FileCheck %s
;
; RUN: opt -passes=sandbox-vectorizer -sbvec-print-pass-pipeline \
; RUN:     -disable-output -sbvec-passes="seed-collection<>,regions-from-metadata<>" %s \
; RUN:     | FileCheck --check-prefix CHECK-MULTIPLE-FUNCTION-PASSES %s

; !!!WARNING!!! This won't get updated by update_test_checks.py !

; This checks the user defined pass pipeline.
define void @pipeline() {
  ret void
}

; CHECK: fpm
; CHECK: seed-collection
; CHECK: rpm
; CHECK: null
; CHECK: null
; CHECK-EMPTY:

; CHECK-MULTIPLE-FUNCTION-PASSES: fpm
; CHECK-MULTIPLE-FUNCTION-PASSES: seed-collection
; CHECK-MULTIPLE-FUNCTION-PASSES: rpm
; CHECK-MULTIPLE-FUNCTION-PASSES: regions-from-metadata
; CHECK-MULTIPLE-FUNCTION-PASSES: rpm
; CHECK-MULTIPLE-FUNCTION-PASSES-EMPTY:
