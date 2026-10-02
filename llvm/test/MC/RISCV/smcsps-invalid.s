# RUN: split-file %s %t
# RUN: not llvm-mc -triple=riscv32 %t/no-features.s 2>&1 \
# RUN:     | FileCheck --check-prefix=NO-FEATURES %t/no-features.s
# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-smcsps \
# RUN:     %t/no-sscsps.s 2>&1 \
# RUN:     | FileCheck --check-prefix=NO-SSCPS %t/no-sscsps.s
# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-sscsps \
# RUN:     %t/invalid-operands.s 2>&1 \
# RUN:     | FileCheck --check-prefix=INVALID-OPERANDS %t/invalid-operands.s

#--- no-features.s
mcspspush sp, sp
# NO-FEATURES: :[[#@LINE-1]]:1: error: instruction requires the following: 'Smcsps'
mcspspop sp, sp
# NO-FEATURES: :[[#@LINE-1]]:1: error: instruction requires the following: 'Smcsps'

#--- no-sscsps.s
scspspush sp, sp
# NO-SSCPS: :[[#@LINE-1]]:1: error: instruction requires the following: 'Sscsps'
scspspop sp, sp
# NO-SSCPS: :[[#@LINE-1]]:1: error: instruction requires the following: 'Sscsps'

#--- invalid-operands.s
mcspspush x1, sp
# INVALID-OPERANDS: :[[#@LINE-1]]:11: error: register must be sp (x2)
mcspspop sp, x1
# INVALID-OPERANDS: :[[#@LINE-1]]:14: error: register must be sp (x2)
scspspush x1, sp
# INVALID-OPERANDS: :[[#@LINE-1]]:11: error: register must be sp (x2)
scspspop sp, x1
# INVALID-OPERANDS: :[[#@LINE-1]]:14: error: register must be sp (x2)
