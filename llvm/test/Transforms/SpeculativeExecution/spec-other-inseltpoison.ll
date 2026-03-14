; RUN: opt < %s -S -passes=speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

; CHECK-LABEL: @ifThen_shuffle(
; CHECK: shufflevector
; CHECK: br i1 true
define void @ifThen_shuffle() {
  br i1 true, label %a, label %b

a:
  %x = shufflevector <2 x float> undef, <2 x float> poison, <2 x i32> zeroinitializer
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_extract(
; CHECK: extractelement
; CHECK: br i1 true
define void @ifThen_extract() {
  br i1 true, label %a, label %b

a:
  %x = extractelement <2 x float> undef, i32 1
  br label %b

b:
  ret void
}


; CHECK-LABEL: @ifThen_insert(
; CHECK: insertelement
; CHECK: br i1 true
define void @ifThen_insert() {
  br i1 true, label %a, label %b

a:
  %x = insertelement <2 x float> poison, float undef, i32 1
  br label %b

b:
  ret void
}

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

; CHECK-LABEL: @ifThen_freeze(
; CHECK: freeze
; CHECK: br i1 true
define void @ifThen_freeze() {
  br i1 true, label %a, label %b

a:
  %x = freeze i32 undef
  br label %b

b:
  ret void
}
