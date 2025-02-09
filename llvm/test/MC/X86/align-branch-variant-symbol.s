# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=call+indirect %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=CHECK,X64
# RUN: llvm-mc -filetype=obj -triple i386 --x86-align-branch-boundary=32 --x86-align-branch=call+indirect %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=CHECK,X86

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
  # X64:    1e:          callq
  # X86:    1e:          calll
  # CHECK:    23:          int3
  call    ___tls_get_addr@PLT
  int3

  .p2align  5
  .rept 30
  int3
  .endr
  # CHECK:    5d:          int3
  # X64:    5e:          callq    *(%ecx)
  # X64:    65:          int3
  # X86:    5e:          calll    *(%ecx)
  # X86:    64:          int3
  call *___tls_get_addr@GOT(%ecx)
  int3

  .p2align  5
  .rept 30
  int3
  .endr
  # CHECK:    9d:          int3
  # X64:    9e:          callq    *(%eax)
  # X64:    a1:          int3
  # X86:    9e:          calll    *(%eax)
  # X86:    a0:          int3
  call *foo@tlscall(%eax)
  int3

  .p2align  5
  .rept 30
  int3
  .endr
  # CHECK:    dd:          int3
  # X64:    de:          jmpq    *(%eax)
  # X64:    e1:          int3
  # X86:    de:          jmpl    *(%eax)
  # X86:    e0:          int3
  jmp  *foo@tlscall(%eax)
  int3
