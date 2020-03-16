# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-pc-linux-gnu %s -x86-pad-max-prefix-size=5 | llvm-objdump -d --section=.text - | FileCheck %s

# This test file highlights the interactions between prefix padding and
# relaxation padding.

  .file "test.c"
  .text
  .section  .text
  # We can both relax and prefix for padding purposes, but the moment, we
  # can't prefix without first having relaxed.
  # CHECK: .text
  # CHECK:  0: eb 1f                         jmp
  # CHECK:  2: eb 1d                         jmp
  # CHECK:  4: eb 1b                         jmp
  # CHECK:  6: eb 19                         jmp
  # CHECK:  8: eb 17                         jmp
  # CHECK:  a: 2e 2e 2e 2e 2e e9 0d 00 00 00 jmp
  # CHECK: 14: 2e 2e 2e 2e 2e e9 03 00 00 00 jmp
  # CHECK: 1e: 66 90                         nop
  # CHECK: 20: cc                            int3
  .p2align 4
  jmp foo
  jmp foo
  jmp foo
  jmp foo
  jmp foo
  jmp foo
  jmp foo
  .p2align 5
  int3
foo:
  ret

  # Canonical toy loop to show benefit - we can align the loop header with
  # fewer nops by relaxing the branch, even though we don't need to
  # CHECK: <loop_preheader>:
  # CHECK: 45: 48 85 c0                       testq %rax, %rax
  # CHECK: 48: 2e 2e 2e 2e 0f 8e 1e 00 00 00  jle 30 <loop_exit>
  # CHECK: 52: 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00    	nopw	%cs:(%rax,%rax)
  # CHECK: <loop_header>:
  # CHECK: 60: 48 83 e8 01                    subq $1, %rax
  # CHECK: 64: 48 85 c0                       testq %rax, %rax
  # CHECK: 67: 7e 07                          jle 7 <loop_exit>
  # CHECK: 69: 2e 2e e9 f0 ff ff ff           jmp
  # CHECK: <loop_exit>:
  # CHECK: 70: c3                             retq
  .p2align 5
  .skip 5
loop_preheader:
  testq %rax, %rax
  jle loop_exit
  .p2align 5
loop_header:
  subq $1, %rax
  testq %rax, %rax
  jle loop_exit
  jmp loop_header
  .p2align 4
loop_exit:
  ret

  # Correctness cornercase - can't prefix pad jmp without having relaxed it
  # first as doing so would make the relative offset too large
  # CHECK: fd: cc                             int3
  # CHECK: fe: eb 80                          jmp -128 <loop_exit+0x10>
  # CHECK: 100: cc                           	int3
.p2align 5
.L1:
.rept 126
  int3
.endr
  jmp .L1
.rept 30
  int3
.endr
.p2align 5
