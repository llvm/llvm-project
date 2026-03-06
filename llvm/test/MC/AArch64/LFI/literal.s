// RUN: not llvm-mc -triple aarch64_lfi %s 2>&1 | FileCheck %s

ldr x0, foo
// CHECK: error: PC-relative literal loads are not supported in LFI

ldr w0, bar
// CHECK: error: PC-relative literal loads are not supported in LFI

ldr s0, baz
// CHECK: error: PC-relative literal loads are not supported in LFI

ldr d0, qux
// CHECK: error: PC-relative literal loads are not supported in LFI

ldr q0, quux
// CHECK: error: PC-relative literal loads are not supported in LFI

ldrsw x0, signed_word
// CHECK: error: PC-relative literal loads are not supported in LFI

foo:
  .quad 0
bar:
  .word 0
baz:
  .single 0.0
qux:
  .double 0.0
quux:
  .zero 16
signed_word:
  .word -1
