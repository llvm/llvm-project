# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=call+jmp+indirect+ret+jcc %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # These tests are checking the basic cases for each instructions, and a
  # bit of the alignment checking logic itself.  Fused instruction cases are
  # excluded, as are details of argument parsing.

  # instruction sizes for reference:
  #  callq is 5 bytes long
  #  int3 is 1 byte
  #  jmp <near-label> is 2 bytes
  #  jmp <far-label> is 5 bytes
  #  ret N is 2 bytes

  # Next couple tests are checking the edge cases on the alignment computation

  .text
  # CHECK: <test1>:
  # CHECK: 20: callq
  .globl  test1
  .p2align  5
test1:
  .rept 29
  int3
  .endr
  callq bar

  # CHECK: <test2>:
  # CHECK: 60: callq
  .globl  test2
  .p2align  5
test2:
  .rept 31
  int3
  .endr
  callq bar

  # CHECK: <test3>:
  # CHECK: a0: callq
  .globl  test3
  .p2align  5
test3:
  .rept 27
  int3
  .endr
  callq bar

  # next couple check instruction type coverage

  # CHECK: <test_jmp>:
  # CHECK: e0: jmp
  .globl  test_jmp
  .p2align  5
test_jmp:
  .rept 31
  int3
  .endr
  jmp bar

  # CHECK: <test_ret>:
  # CHECK: 120: retq
  .globl  test_ret
  .p2align  5
test_ret:
  .rept 31
  int3
  .endr
  retq $0

  # check a case with a relaxable instruction

  # CHECK: <test_jmp_far>:
  # CHECK: 160: jmp
  .globl  test_jmp_far
  .p2align  5
test_jmp_far:
  .rept 31
  int3
  .endr
  jmp baz

  # CHECK: <test_jcc>:
  # CHECK: 1a0: jne
  .globl  test_jcc
  .p2align  5
test_jcc:
  .rept 31
  int3
  .endr
  jne bar

  # CHECK: <test_indirect>:
  # CHECK: 1e0: jmp
  .globl  test_indirect
  .p2align  5
test_indirect:
  .rept 31
  int3
  .endr
  jmpq *(%rax)

  .p2align 4
  .type   bar,@function
bar:
  retq

  # This case looks really tempting to pad, but doing so for the call causes
  # the jmp to be misaligned.
  # CHECK: <test_pad_via_relax_neg1>:
  # CHECK: 200: int3
  # CHECK: 21a: testq
  # CHECK: 21d: jne
  # CHECK: 21f: nop
  # CHECK: 220: callq
  .global test_pad_via_relax_neg1
  .p2align  5
test_pad_via_relax_neg1:
  .rept 26
  int3
  .endr
  testq %rax, %rax
  jnz bar
  callq bar

  # Same as previous, but without fusion
  # CHECK: <test_pad_via_relax_neg2>:
  # CHECK: 240: int3
  # CHECK: 25d: jmp
  # CHECK: 25f: nop
  # CHECK: 260: callq
  .global test_pad_via_relax_neg2
  .p2align  5
test_pad_via_relax_neg2:
  .rept 29
  int3
  .endr
  jmp bar2
  callq bar2

bar2:

  .section "unknown"
  .p2align 4
  .type   baz,@function
baz:
  retq
