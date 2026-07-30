; RUN: llc < %s -code-model=large -mcpu=core2 -mtriple=x86_64-- -O0 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@.str10 = external unnamed_addr constant [2 x i8], align 1

define void @foo() {
; CHECK-LABEL: foo:
entry:
; CHECK: callq
  %call = call ptr undef(ptr undef, ptr @.str10)
  ret void
}
