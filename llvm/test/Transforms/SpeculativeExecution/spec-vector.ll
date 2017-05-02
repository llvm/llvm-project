; RUN: opt < %s -S -speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

; CHECK-LABEL: @ifThen_extractelement_constindex(
; CHECK: extractelement
; CHECK: br i1 true
define void @ifThen_extractelement_constindex() {
  br i1 true, label %a, label %b

a:
  %x = extractelement <4 x i32> undef, i32 0
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_extractelement_varindex(
; CHECK: extractelement
; CHECK: br i1 true
define void @ifThen_extractelement_varindex(i32 %idx) {
  br i1 true, label %a, label %b

a:
  %x = extractelement <4 x i32> undef, i32 %idx
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_insertelement_constindex(
; CHECK: insertelement
; CHECK: br i1 true
define void @ifThen_insertelement_constindex() {
  br i1 true, label %a, label %b

a:
  %x = insertelement <4 x i32> undef, i32 undef, i32 0
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_insertelement_varindex(
; CHECK: insertelement
; CHECK: br i1 true
define void @ifThen_insertelement_varindex(i32 %idx) {
  br i1 true, label %a, label %b

a:
  %x = insertelement <4 x i32> undef, i32 undef, i32 %idx
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_shufflevector(
; CHECK: shufflevector
; CHECK: br i1 true
define void @ifThen_shufflevector() {
  br i1 true, label %a, label %b

a:
  %x = shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> undef
  br label %b

b:
  ret void
}
