; RUN: opt %s -passes=aa-eval -disable-output -print-all-alias-modref-info 2>&1 | FileCheck %s

; CHECK-LABEL: Function: patatino
; CHECK: NoAlias: ptr* %G22, ptr* %G45

define void @patatino() {
BB:
  %G22 = getelementptr ptr, ptr undef, i8 -1
  %B1 = mul i66 undef, 9223372036854775808
  %G45 = getelementptr ptr, ptr undef, i66 %B1
  load ptr, ptr %G22
  load ptr, ptr %G45
  ret void
}
