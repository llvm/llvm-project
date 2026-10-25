# RUN: split-file %s %t
# RUN: not llvm-mc -triple=riscv32 %t/no-features.s 2>&1 \
# RUN:     | FileCheck --check-prefix=NO-FEATURES %t/no-features.s
# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-smip \
# RUN:     %t/no-ssip.s 2>&1 \
# RUN:     | FileCheck --check-prefix=NO-SSIP %t/no-ssip.s
# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-ssip \
# RUN:     %t/no-smip.s 2>&1 \
# RUN:     | FileCheck --check-prefix=NO-SMIP %t/no-smip.s
# RUN: not llvm-mc -triple=riscv32 \
# RUN:     -mattr=+experimental-smip,+experimental-ssip \
# RUN:     %t/invalid-operands.s 2>&1 \
# RUN:     | FileCheck --check-prefix=INVALID-OPERANDS %t/invalid-operands.s

#--- no-features.s
mipopret
# NO-FEATURES: :[[#@LINE-1]]:1: error: instruction requires the following: 'Smip'
sipopret
# NO-FEATURES: :[[#@LINE-1]]:1: error: instruction requires the following: 'Ssip'

#--- no-ssip.s
sipopret
# NO-SSIP: :[[#@LINE-1]]:1: error: instruction requires the following: 'Ssip'

#--- no-smip.s
mipopret
# NO-SMIP: :[[#@LINE-1]]:1: error: instruction requires the following: 'Smip'

#--- invalid-operands.s
mipopret zero
# INVALID-OPERANDS: :[[#@LINE-1]]:10: error: unexpected extra operand for instruction
sipopret zero
# INVALID-OPERANDS: :[[#@LINE-1]]:10: error: unexpected extra operand for instruction
