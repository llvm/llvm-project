; RUN: opt < %s -S -speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

; CHECK-LABEL: @ifThen_icmp(
; CHECK: icmp
; CHECK: br i1 true
define void @ifThen_icmp() {
  br i1 true, label %a, label %b

a:
  %x = icmp eq i32 undef, undef
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fcmp(
; CHECK: fcmp
; CHECK: br i1 true
define void @ifThen_fcmp() {
  br i1 true, label %a, label %b

a:
  %x = fcmp oeq float undef, undef
  br label %b

b:
  ret void
}
