# RUN: not llvm-mc -triple mips-unknown-linux %s 2>&1 | FileCheck %s

# CHECK: :[[#@LINE+1]]:11: error: expected string
  .asciiz 12
# CHECK: :[[#@LINE+1]]:14: error: unexpected token
  .asciiz "a"3
# CHECK: :[[#@LINE+1]]:15: error: expected string
  .asciiz "a",
