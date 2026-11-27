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

// CHECK: error: scale factor in address must be 1, 2, 4 or 8
  xor [rsi + rbx*(6-6)], eax
// CHECK: error: scale factor in address must be 1, 2, 4 or 8
  xor [rsi + rbx*(4+5)], eax
// CHECK: error: Scale can't be negative
  xor [rsi + rbx*(4-8)], eax
// CHECK: error: unmatched parenthesis
  xor [rsi + rbx*(1+1], eax
// CHECK: error: expected absolute value
  xor [rsi + rbx*(unknown_scale)], eax
// CHECK: error: scale factor in address must be 1, 2, 4 or 8
  xor [rsi + rbx*(0x100000000+2)], eax
// CHECK: error: division by zero in assembly expression
  xor [rsi + rbx*(64/(2-2))], eax
// CHECK: error: modulo by zero in assembly expression
  xor [rsi + rbx*(64%(2-2))], eax
// CHECK: error: division result out of range
  xor [rsi + rbx*(-0x8000000000000000/-1)], eax
// CHECK: error: invalid shift count in assembly expression
  xor [rsi + rbx*(1<<64)], eax
