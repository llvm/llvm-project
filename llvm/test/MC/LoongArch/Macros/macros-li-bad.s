# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

li.w $a0, 0x100000000
# CHECK: :[[#@LINE-1]]:11: error: operand must be a 32 bit immediate

li.d $a0, 0x10000000000000000
# CHECK: :[[#@LINE-1]]:11: error: unknown operand
