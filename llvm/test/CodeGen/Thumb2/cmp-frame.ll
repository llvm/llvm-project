; RUN: llc < %s | FileCheck %s

target triple = "thumbv7-linux-androideabi"

define i1 @f() {
  %a = alloca ptr
  ; CHECK: adds.w r0, sp, #0
  ; CHECK: it ne
  %cmp = icmp ne ptr %a, null
  ret i1 %cmp
}
