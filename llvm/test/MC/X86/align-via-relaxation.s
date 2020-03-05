# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-pc-linux-gnu %s | llvm-objdump -d --section=.text - | FileCheck %s


  .file "test.c"
  .text
  .section  .text
  # Demonstrate that we can relax instructions to provide padding, not
  # just insert nops.  jmps are being used for ease of demonstration.
  # CHECK: .text
  # CHECK: 0: eb 1f                         jmp 31 <foo>
  # CHECK: 2: e9 1a 00 00 00                jmp 26 <foo>
  # CHECK: 7: e9 15 00 00 00                jmp 21 <foo>
  # CHECK: c: e9 10 00 00 00                jmp 16 <foo>
  # CHECK: 11: e9 0b 00 00 00               jmp 11 <foo>
  # CHECK: 16: e9 06 00 00 00               jmp 6 <foo>
  # CHECK: 1b: e9 01 00 00 00               jmp 1 <foo>
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
  # CHECK: bar:
  # CHECK: 22: eb fe                          jmp -2 <bar>
  # CHECK: 24: 66 2e 0f 1f 84 00 00 00 00 00  nopw %cs:(%rax,%rax)
  # CHECK: 2e: 66 90                          nop
  # CHECK: 30: 0f 0b                          ud2

bar:  
  jmp bar
nobypass:
  .p2align 4
  ud2


  # Canonical toy loop to show benefit - we can align the loop header with
  # fewer nops by relaxing the branch, even though we don't need to
  # CHECK: loop_preheader:
  # CHECK: 45: 48 85 c0                       testq %rax, %rax
  # CHECK: 48: 0f 8e 22 00 00 00              jle 34 <loop_exit>
  # CHECK: 4e: 66 2e 0f 1f 84 00 00 00 00 00  nopw %cs:(%rax,%rax)
  # CHECK: 58: 0f 1f 84 00 00 00 00 00        nopl (%rax,%rax)
  # CHECK: loop_header:
  # CHECK: 60: 48 83 e8 01                    subq $1, %rax
  # CHECK: 64: 48 85 c0                       testq %rax, %rax
  # CHECK: 67: 7e 07                          jle 7 <loop_exit>
  # CHECK: 69: e9 f2 ff ff ff                 jmp -14 <loop_header>
  # CHECK: 6e: 66 90                          nop
  # CHECK: loop_exit:
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
