  # RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-pc-linux-gnu %s -x86-pad-max-prefix-size=1 --x86-align-branch-boundary=32 --x86-align-branch=jmp+indirect | llvm-objdump -d - | FileCheck %s
  # RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-pc-linux-gnu %s --mc-relax-all | llvm-objdump -d - | FileCheck --check-prefixes=RELAX-ALL %s

  # Exercise cases where we are allowed to increase the length of unrelaxable
  # instructions (by adding prefixes) for alignment purposes.

  # The first test checks instructions 'int3', 'push %rbp', which will be padded
  # later are unrelaxable (their encoding size is still 1 byte when
  # --mc-relax-all is passed).
  .text
  .globl labeled_unrelaxable_test
labeled_unrelaxable_test:
# RELAX-ALL:       0: cc                               int3
# RELAX-ALL:       1: 54                               pushq    %rsp
  int3
  push %rsp

  # The second test is a basic test, we just check the jmp is aligned by prefix
  # padding the previous instructions.
  .text
  .globl labeled_basic_test
labeled_basic_test:
  .p2align 5
  .rept 28
  int3
  .endr
# CHECK:      3c: 2e cc                            int3
# CHECK:      3e: 2e 54                            pushq    %rsp
# CHECK:      40: eb 00                            jmp
  int3
  push %rsp
  jmp foo
foo:
  ret

   # The third test check the correctness cornercase - can't add prefixes on a
   # prefix or a instruction following by a prefix.
  .globl labeled_prefix_test
labeled_prefix_test:
  .p2align 5
  .rept 28
  int3
  .endr
# CHECK:      7c: 2e cc                            int3
  int3
# CHECK:      7e: 3e cc                            int3
  DS
  int3
# CHECK:      80: eb 00                            jmp
  jmp bar
bar:
  ret
