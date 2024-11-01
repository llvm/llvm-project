# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

la $a0, $a1, sym
# CHECK: :[[#@LINE-1]]:10: error: operand must be a bare symbol name

la $a0, 1
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name

la.global $a0, $a1, 1
# CHECK: :[[#@LINE-1]]:21: error: operand must be a bare symbol name
