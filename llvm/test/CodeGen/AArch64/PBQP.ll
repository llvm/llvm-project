; RUN: llc -mtriple=aarch64-linux-gnu -mcpu=cortex-a57 -regalloc=pbqp -pbqp-coalescing -o - %s | FileCheck %s

define i32 @foo(i32 %a) {
; CHECK-LABEL: foo:
; CHECK: bl bar
; CHECK: bl baz
  %call = call i32 @bar(i32 %a)
  %call1 = call i32 @baz(i32 %call)
  ret i32 %call1
}

declare i32 @bar(i32)
declare i32 @baz(i32)

