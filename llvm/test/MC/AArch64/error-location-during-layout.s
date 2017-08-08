// RUN: not llvm-mc -triple aarch64--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  .section a
  .space 8
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid .org offset '4' (at offset '8')
  .org 4

  .section b
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected absolute expression
  .org undef

  .section c
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected assembly-time absolute expression
  .org -undef
