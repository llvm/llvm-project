# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec %t -show-graphs='.*' | FileCheck %s

# CHECK: linkage: weak, scope: local, live  -   _foo_weak

# _foo_weak is weak and local. Make sure we can link it.
  .section  __TEXT,__text,regular,pure_instructions
  .weak_definition  _foo_weak
  .p2align  4, 0x90
_foo_weak:
  retq

  .globl  _main
  .p2align  4, 0x90
_main:
  jmp _foo_weak

  .subsections_via_symbols
