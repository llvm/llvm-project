; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s
declare void @bar(ptr, ptr, ptr)

; SelectionDAG used to try to fold some fp128 operations using the ppc128 type,
; which is not supported.

define fp128 @test_folding() {
; CHECK-LABEL: test_folding:
  %l = alloca i32
  store i32 42, ptr %l
  %val = load i32, ptr %l
  %fpval = sitofp i32 %val to fp128
  ; If the value is loaded from a constant pool into an fp128, it's been folded
  ; successfully.
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.LCPI
  ret fp128 %fpval
}
