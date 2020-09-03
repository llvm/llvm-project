# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=jcc %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=JCC 
# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=jmp %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=JMP
# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=indirect %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=IND
# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=call %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=CAL
# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=ret %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=RET

  # Exercise cases where only one kind of instruction is asked to be aligned.
  # Fused instruction cases are excluded.

  .text
  .globl  foo
  .p2align  5
foo:
  .p2align  5
  .rept 30
  int3
  .endr
  # JCC:    20:          jne
  # JMP:    1e:          jne
  # IND:    1e:          jne
  # CAL:    1e:          jne
  # RET:    1e:          jne
  jne foo
  int3

  .p2align  5
  .rept 30
  int3
  .endr
  # JCC:    5e:          jmp
  # JMP:    60:          jmp
  # IND:    5e:          jmp
  # CAL:    5e:          jmp
  # RET:    5e:          jmp
  jmp foo
  int3

  .p2align  5
  .rept 30
  int3
  .endr
  # JCC:    9e:          jmpq    *%rax
  # JMP:    9e:          jmpq    *%rax
  # IND:    a0:          jmpq    *%rax
  # CAL:    9e:          jmpq    *%rax
  # RET:    9e:          jmpq    *%rax
  jmp  *%rax
  int3


  .p2align  5
  .rept 30
  int3
  .endr
  # JCC:    de:          callq    *%rax
  # JMP:    de:          callq    *%rax
  # IND:    de:          callq    *%rax
  # CAL:    e0:          callq    *%rax
  # RET:    de:          callq    *%rax
  call  *%rax
  int3


  .p2align  5
  .rept 30
  int3
  .endr
  # JCC:   11e:          retq
  # JMP:   11e:          retq
  # IND:   11e:          retq
  # CAL:   11e:          retq
  # RET:   120:          retq
  ret $0
  int3


  .p2align  5
  .rept 29
  int3
  .endr
  # JCC:   15d:          cmpq    %rax, %rbp
  # JCC:   160:          je
  # JMP:   15d:          cmpq    %rax, %rbp
  # JMP:   160:          je
  # IND:   15d:          cmpq    %rax, %rbp
  # IND:   160:          je
  # CAL:   15d:          cmpq    %rax, %rbp
  # CAL:   160:          je
  # RET:   15d:          cmpq    %rax, %rbp
  # RET:   160:          je
  cmp  %rax, %rbp
  je  foo
  int3
