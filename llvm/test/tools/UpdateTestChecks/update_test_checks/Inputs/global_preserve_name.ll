; RUN: opt -S < %s | FileCheck %s

@G = constant i32 42

;.
; CHECK: @G = constant i32 42
;.
define ptr @foo() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    ret ptr @G
;
  ret ptr @G
}
