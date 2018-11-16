; RUN: not llvm-mc -triple msp430 < %s 2>&1 | FileCheck %s
foo:
  ;; invalid operand count
  mov    r7        ; CHECK: :[[@LINE]]:3: error: too few operands for instruction

  ;; invalid destination addressing modes
  mov    r7, @r15  ; CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  mov    r7, @r15+ ; CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  mov    r7, #0    ; CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  mov    r7, #123  ; CHECK: :[[@LINE]]:14: error: invalid operand for instruction

  ;; invalid byte instructions
  swpb.b r7        ; CHECK: :[[@LINE]]:3: error: invalid instruction mnemonic
  sxt.b  r7        ; CHECK: :[[@LINE]]:3: error: invalid instruction mnemonic
  call.b r7        ; CHECK: :[[@LINE]]:3: error: invalid instruction mnemonic

  ;; invalid conditional jump offsets
  jmp    -513      ; CHECK: :[[@LINE]]:10: error: invalid jump offset
  jmp    512       ; CHECK: :[[@LINE]]:10: error: invalid jump offset
