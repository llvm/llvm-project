// RUN: not llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s 2>&1 | FileCheck %s

// CHECK: error: Scale can't be negative
  xor [rsi - (rdx) + 40], eax
// CHECK: error: Scale can't be negative
  xor [rsi - 8*(rdx)], eax
// CHECK: error: BaseReg/IndexReg already set!
  xor [rsi + (4*rbx) + rbx], eax
// CHECK: error: Register can't be multiplied with register!
  xor [rsi + (rax*rbx)], eax
// CHECK: error: Register can't be multiplied with register!
  xor [rsi + rax*(rbx)], eax
// CHECK: error: scale factor in address must be 1, 2, 4 or 8
  xor [rsi + 0*(rbx)], eax
// CHECK:  error: unmatched parenthesis
  xor [rsi + 2*(rbx + 23], eax
// CHECK:  error: unmatched parenthesis
  xor [rsi + 2*(rbx)) + 23], eax
