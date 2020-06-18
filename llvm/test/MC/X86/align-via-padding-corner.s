  # RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-pc-linux-gnu %s -x86-pad-max-prefix-size=5 | llvm-objdump -d - | FileCheck %s


  # The first test check the correctness cornercase - can't add prefixes on a
  # instruction following by a prefix.
  .globl labeled_prefix_test
labeled_prefix_test:
# CHECK:       0: 2e 2e 2e 2e 2e e9 06 00 00 00    jmp
# CHECK:       a: 3e e9 00 00 00 00                jmp
  jmp bar
  DS
  jmp bar
  .p2align 4
bar:
  ret

  # The second test is similar to the second test - can't add prefixes on a
  # instruction following by hardcode.
  .p2align 5
  .globl labeled_hardcode_test
labeled_hardcode_test:
# CHECK:      20: 2e 2e 2e 2e 2e e9 06 00 00 00    jmp
# CHECK:      2a: 3e e9 00 00 00 00                jmp
  jmp baz
  .byte 0x3e
  jmp baz
  .p2align 4
baz:
  ret
