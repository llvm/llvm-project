; RUN: llc -mtriple=x86_64 -o - %s | FileCheck %s

@var = global [30 x i64] zeroinitializer

define graalcc void @keep_live() {
  %val = load volatile [30 x i64], ptr @var
  store volatile [30 x i64] %val, ptr @var
; CHECK-NOT: movq {{[0-9]+}}(%{{[a-z]+}}), %r14
; CHECK-NOT: movq {{[0-9]+}}(%{{[a-z]+}}), %r15
; CHECK-NOT: movq %r14,
; CHECK-NOT: movq %r15,
  ret void
}
