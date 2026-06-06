# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

la.got $a0, 1
# CHECK: :[[#@LINE-1]]:13: error: operand must be a bare symbol name

la.pcrel $a0, $a1, 1
# CHECK: :[[#@LINE-1]]:20: error: operand must be a bare symbol name

la.abs $a0, $a1, sym
# CHECK: :[[#@LINE-1]]:14: error: operand must be a bare symbol name

la.pcrel $a0, $a0, sym
# CHECK: :[[#@LINE-1]]:11: error: $rd must be different from $rj

la.tls.desc $a1, sym
# CHECK: :[[#@LINE-1]]:14: error: $rd must be $r4
