  # RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --x86-align-branch-boundary=32 --x86-align-branch=jmp+call %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # Exercise cases where prefixes are specified for instructions to be aligned
  # and thus can't add a nop in between without changing semantic.

  .text

  # CHECK: 1d:       int3
  # CHECK: 1e:       jmp
  # CHECK: 24:       int3
  .p2align  5
  .rept 30
  int3
  .endr
  CS
  jmp baz
  int3

  # CHECK: 5d:       int3
  # CHECK: 5e:       jmp
  # CHECK: 64:       int3
  .p2align  5
  .rept 30
  int3
  .endr
  GS
  jmp baz
  int3

  # CHECK: 9d:       int3
  # CHECK: 9e:       call
  # CHECK: a6:       int3
  .p2align  5
  .rept 30
  int3
  .endr
  data16
  call *___tls_get_addr@GOT(%ecx)
  int3

  # CHECK: de:       lock
  # CHECK: df:       jmp
  # CHECK: e4:       int3
  .p2align  5
  .rept 30
  int3
  .endr
  lock
  jmp baz
  int3

  # CHECK: 11d:       int3
  # CHECK: 11e:       jmp
  # CHECK: 124:       int3
  .p2align  5
  .rept 30
  int3
  .endr
  rex64
  jmp baz
  int3

  # CHECK: 15d:      int3
  # CHECK: 15e:      {{.*}} jmp
  # CHECK: 164:      int3
  .p2align  5
  .rept 30
  int3
  .endr
  xacquire
  jmp baz
  int3

  .section ".text.other"
bar:
  retq
