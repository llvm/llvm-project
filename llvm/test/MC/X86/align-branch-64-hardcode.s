  # RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --x86-align-branch-boundary=32 --x86-align-branch=jmp+call %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # Exercise cases where instructions to be aligned is after hardcode
  # and thus can't add a nop in between without changing semantic.

  .text

  # CHECK: 1d:       int3
  # CHECK: 1e:       jmp
  # CHECK: 24:       int3
  .p2align  5
  .rept 30
  int3
  .endr
  .byte 0x2e
  jmp baz
  int3

  # CHECK: 5d:       int3
  # CHECK: 5e:       call
  # CHECK: 66:       int3
  .p2align  5
  .rept 30
  int3
  .endr
  .byte 0x66
  call *___tls_get_addr@GOT(%ecx)
  int3

  .section ".text.other"
bar:
  retq
