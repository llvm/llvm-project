; RUN: opt < %s -S -speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

; CHECK-LABEL: @ifThen_fadd(
; CHECK: fadd
; CHECK: br i1 true
define void @ifThen_fadd() {
  br i1 true, label %a, label %b

a:
  %x = fadd float undef, undef
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fsub(
; CHECK: fsub
; CHECK: br i1 true
define void @ifThen_fsub() {
  br i1 true, label %a, label %b

a:
  %x = fsub float undef, undef
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fneg(
; CHECK: fsub float -0.0
; CHECK: br i1 true
define void @ifThen_fneg() {
  br i1 true, label %a, label %b

a:
  %x = fsub float -0.0, undef
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fmul(
; CHECK: fmul
; CHECK: br i1 true
define void @ifThen_fmul() {
  br i1 true, label %a, label %b

a:
  %x = fmul float undef, undef
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fdiv(
; CHECK: fdiv
; CHECK: br i1 true
define void @ifThen_fdiv() {
  br i1 true, label %a, label %b

a:
  %x = fdiv float undef, undef
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_frem(
; CHECK: frem
; CHECK: br i1 true
define void @ifThen_frem() {
  br i1 true, label %a, label %b

a:
  %x = frem float undef, undef
  br label %b

b:
  ret void
}
