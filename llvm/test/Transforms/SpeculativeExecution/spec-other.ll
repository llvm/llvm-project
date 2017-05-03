; RUN: opt < %s -S -speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

; CHECK-LABEL: @ifThen_extractvalue(
; CHECK: extractvalue
; CHECK: br i1 true
define void @ifThen_extractvalue() {
  br i1 true, label %a, label %b

a:
  %x = extractvalue { i32, i32 } undef, 0
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_insertvalue(
; CHECK: insertvalue
; CHECK: br i1 true
define void @ifThen_insertvalue() {
  br i1 true, label %a, label %b

a:
  %x = insertvalue { i32, i32 } undef, i32 undef, 0
  br label %b

b:
  ret void
}

