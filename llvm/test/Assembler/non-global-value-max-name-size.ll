; RUN: opt < %s -S -non-global-value-max-name-size=5
; RUN: not opt < %s -S -non-global-value-max-name-size=4 2>&1 | FileCheck %s

; CHECK: name is too long

define void @f() {
bb0:
  br label %testz

testz:
  br label %testa

testa:
  br label %testz
}
