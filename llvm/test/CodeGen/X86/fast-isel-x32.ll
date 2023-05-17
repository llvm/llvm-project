; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -fast-isel -fast-isel-abort=1 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-nacl -fast-isel -fast-isel-abort=1 | FileCheck %s

; Test that alloca addresses are materialized with the right size instruction.

declare void @bar(ptr %arg)

; CHECK-LABEL: @foo
define void @foo() {
  %a = alloca i32
; CHECK: leal {{.*}}, %edi
  call void @bar(ptr %a)
  ret void
}
