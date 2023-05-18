# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-pc-linux-gnu -x86-pad-max-prefix-size=0 %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=NOPAD
# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-pc-linux-gnu -x86-pad-max-prefix-size=0 -x86-pad-for-align=1 %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

# This test exercises only the padding via relaxation logic.  The  interaction
# etween prefix padding and relaxation logic can be seen in align-via-padding.s

  .file "test.c"
  .text
  .section  .text

# NOPAD-LABEL: <.text>:
# NOPAD-NEXT:     0: eb 1f           jmp 0x21 <foo>
# NOPAD-NEXT:     2: eb 1d           jmp 0x21 <foo>
# NOPAD-NEXT:     4: eb 1b           jmp 0x21 <foo>
# NOPAD-NEXT:     6: eb 19           jmp 0x21 <foo>
# NOPAD-NEXT:     8: eb 17           jmp 0x21 <foo>
# NOPAD-NEXT:     a: eb 15           jmp 0x21 <foo>
# NOPAD-NEXT:     c: eb 13           jmp 0x21 <foo>
# NOPAD-NEXT:     e: 66 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00  nopw    %cs:(%rax,%rax)
# NOPAD-NEXT:    1d: 0f 1f 00        nopl (%rax)
# NOPAD-NEXT:    20: cc              int3

  # Demonstrate that we can relax instructions to provide padding, not
  # just insert nops.  jmps are being used for ease of demonstration.
  # CHECK: .text
  # CHECK: 0: eb 1f                         jmp 0x21 <foo>
  # CHECK: 2: e9 1a 00 00 00                jmp 0x21 <foo>
  # CHECK: 7: e9 15 00 00 00                jmp 0x21 <foo>
  # CHECK: c: e9 10 00 00 00                jmp 0x21 <foo>
  # CHECK: 11: e9 0b 00 00 00               jmp 0x21 <foo>
  # CHECK: 16: e9 06 00 00 00               jmp 0x21 <foo>
  # CHECK: 1b: e9 01 00 00 00               jmp 0x21 <foo>
  # CHECK: 20: cc                           int3
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

  # Check that we're not shifting aroudn the offsets of labels - doing
  # that would require a further round of relaxation
  # CHECK: <bar>:
  # CHECK: 22: eb fe                          jmp 0x22 <bar>
  # CHECK: 24: 66 66 66 2e 0f 1f 84 00 00 00 00 00 nopw %cs:(%rax,%rax)
  # CHECK: 30: 0f 0b                          ud2

bar:  
  jmp bar
nobypass:
  .p2align 4
  ud2


  # Canonical toy loop to show benefit - we can align the loop header with
  # fewer nops by relaxing the branch, even though we don't need to
  # CHECK: <loop_preheader>:
  # CHECK: 45: 48 85 c0                       testq %rax, %rax
  # CHECK: 48: 0f 8e 22 00 00 00              jle 0x70 <loop_exit>
  # CHECK: 4e: 66 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00 nopw %cs:(%rax,%rax)
  # CHECK: 5d: 0f 1f 00                       nopl (%rax)
  # CHECK: <loop_header>:
  # CHECK: 60: 48 83 e8 01                    subq $1, %rax
  # CHECK: 64: 48 85 c0                       testq %rax, %rax
  # CHECK: 67: 7e 07                          jle 0x70 <loop_exit>
  # CHECK: 69: e9 f2 ff ff ff                 jmp 0x60 <loop_header>
  # CHECK: 6e: 66 90                          nop
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
