; RUN: llc -mtriple=arm64 -o - %s | FileCheck %s

@var = global [30 x i64] zeroinitializer

define graalcc void @keep_live() {
  %val = load volatile [30 x i64], ptr @var
  store volatile [30 x i64] %val, ptr @var
; CHECK-NOT: ldr x27
; CHECK-NOT: ldr x28
  ret void
}
