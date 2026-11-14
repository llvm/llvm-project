// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel --output-asm-variant=1 %s | FileCheck %s

// Test that parentheses around scale register are interpreted correctly
// Tests parentheses around registers and scale values in register addressing
// CHECK: xor dword ptr [rsi + 16], eax
  xor [rsi + (2) * 8], eax
// CHECK: xor dword ptr [rsi + rbx + 16], eax
  xor [rsi + 2 * 8 + (rbx)], eax
// CHECK: xor dword ptr [rsi + 8*rbx + 16], eax
  xor [(rsi) + 2 * 8 + (rbx) * 8], eax
// CHECK: xor dword ptr [rsi + 2*rbx], eax
  xor [rsi + 2*(rbx)], eax
// CHECK: xor dword ptr [rsi + 2*rbx - 40], eax
  xor [rsi - 40 + (2*rbx)], eax
// CHECK: xor dword ptr [rsi + 4*rbx + 24], eax
  xor [rsi + (rbx*2)*2 + 24], eax
// CHECK: xor dword ptr [rsi + 8*rbx], eax
  xor [rsi + 2*(2*2*(rbx))], eax
// CHECK: xor dword ptr [rsi + 8*rbx - 40], eax
  xor [rsi -40 +(2*(1*(rbx*2))*(2))], eax

// CHECK: xor dword ptr [rsi + 2*rbx], eax
  xor [rsi + rbx*(1+1)], eax
// CHECK: xor dword ptr [rsi + 2*rbx + 4], eax
  xor [rsi + rbx*(8-6) + 4], eax
// CHECK: xor dword ptr [rsi + 2*rbx], eax
  xor [rsi + rbx*(4/2)], eax
// CHECK: xor dword ptr [rsi + 4*rbx], eax
  xor [rsi + rbx*((3-1)*2)], eax
// CHECK: xor dword ptr [rsi + 8*rbx], eax
  xor [rsi + rbx*(1<<3)], eax
.set scale_value, 1
// CHECK: xor dword ptr [rsi + 2*rbx], eax
  xor [rsi + rbx*(scale_value+1)], eax
// CHECK: xor dword ptr [rsi + 4*rbx], eax
  xor [rsi + 2*(rbx*(1+1))], eax
// CHECK: xor dword ptr [rsi + rbx], eax
  xor [rsi + rbx*(1 or 1)], eax
// CHECK: xor dword ptr [rsi + rbx], eax
  xor [rsi + rbx*(1 and 1)], eax
// CHECK: xor dword ptr [rsi + 2*rbx], eax
  xor [rsi + rbx*(1 shl 1)], eax
// CHECK: xor dword ptr [rsi + rbx], eax
  xor [rsi + rbx*(2 shr 1)], eax
