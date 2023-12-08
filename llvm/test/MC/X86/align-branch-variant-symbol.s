# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=call+indirect %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=64BIT,CHECK

# RUN: llvm-mc -filetype=obj -triple i386 --x86-align-branch-boundary=32 --x86-align-branch=call+indirect %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=32BIT,CHECK

  # Exercise cases where the instruction to be aligned has a variant symbol
  # operand, and we can't add before it since linker may rewrite it.

  .text
  .global foo

foo:
  .p2align  5
  .rept 30
  int3
  .endr
  # CHECK:    1d:          int3
  # 64BIT:    1e:          callq
  # 32BIT:    1e:          calll
  # CHECK:    23:          int3
  call    ___tls_get_addr@PLT
  int3

  .p2align  5
  .rept 30
  int3
  .endr
  # CHECK:    5d:          int3
  # 64BIT:    5e:          callq    *(%ecx)
  # 64BIT:    65:          int3
  # 32BIT:    5e:          calll    *(%ecx)
  # 32BIT:    64:          int3
  call *___tls_get_addr@GOT(%ecx)
  int3

  .p2align  5
  .rept 30
  int3
  .endr
  # CHECK:    9d:          int3
  # 64BIT:    9e:          callq    *(%eax)
  # 64BIT:    a1:          int3
  # 32BIT:    9e:          calll    *(%eax)
  # 32BIT:    a0:          int3
  call *foo@tlscall(%eax)
  int3

  .p2align  5
  .rept 30
  int3
  .endr
  # CHECK:    dd:          int3
  # 64BIT:    de:          jmpq    *(%eax)
  # 64BIT:    e1:          int3
  # 32BIT:    de:          jmpl    *(%eax)
  # 32BIT:    e0:          int3
  jmp  *foo@tlscall(%eax)
  int3
