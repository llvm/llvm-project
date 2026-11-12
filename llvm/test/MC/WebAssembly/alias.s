# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s | llvm-objdump -t - | FileCheck %s

.section    .data,"",@
foo:
  .int32 1
  .size foo, 4
sym_a:
  .int32 2
  .size sym_a, 4

.set sym_b, sym_a

# CHECK: 00000000 l     O DATA 00000004 foo
# CHECK: 00000004 l     O DATA 00000004 sym_a
# CHECK: 00000004 l     O DATA 00000004 sym_b
