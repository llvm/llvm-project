  # RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --x86-align-branch-boundary=32 --x86-align-branch=ret %s | llvm-readobj -S | FileCheck %s

  # Check the aligment of section that contains instructions to be aligned
  # is correctly set.

  # CHECK:  Name: text1
  # CHECK:  AddressAlignment: 32
  .section text1
foo:
  ret

  # CHECK:  Name: text2
  # CHECK:  AddressAlignment: 1
  .section text2
  nop

  # CHECK:  Name: text3
  # CHECK:  AddressAlignment: 1
  .section text3
  jmp foo
