# RUN: llvm-mc -g -triple i686-linux-gnu -filetype asm -o - %s | FileCheck %s

# 1 "reduced.S"
# 1 "<built-in>" 1
# 1 "reduced.S" 2
# 200 "macros.h"

 .macro return arg
  movl %eax, \arg
  retl
 .endm

 .macro return2 arg
  return \arg
 .endm

# 7 "reduced.S"
function:
 return 0

# CHECK: .file 1 "reduced.S"
# CHECK: .loc 1 8 0
# CHECK: movl %eax, 0
# CHECK: .loc 1 8 0
# CHECK: retl

# 42 "reduced.S"
function2:
 return2 0

# CHECK: .loc 1 43 0
# CHECK: movl %eax, 0
# CHECK: .loc 1 43 0
# CHECK: retl
