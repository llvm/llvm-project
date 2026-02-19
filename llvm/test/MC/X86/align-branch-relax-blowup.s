// REQUIRES: asserts
// Test that sections containing large amounts of branches with alignment
// constraints do not cause quadratic relaxation blowup.

// RUN: llvm-mc -filetype=obj -triple x86_64 --stats -o %t \
// RUN:   --x86-align-branch-boundary=8 --x86-align-branch=call %s 2>&1 \
// RUN: | FileCheck %s
// CHECK: 1 assembler         - Number of assembler layout and relaxation steps

// RUN: llvm-objdump -d --no-show-raw-insn %t \
// RUN: | FileCheck %s --check-prefix=DISASM
// DISASM:  0: callq
// DISASM:  8: callq
// DISASM: 10: callq
// DISASM: 18: callq
// DISASM: 20: callq
// DISASM: 28: callq

  .text
foo:
  callq bar
  callq bar
  callq bar
  callq bar
  .p2align 3
  callq bar
  callq bar
  ret
