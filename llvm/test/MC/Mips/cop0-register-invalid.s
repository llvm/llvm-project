# RUN: not llvm-mc -triple=mips -mcpu=mips32r2 %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=mips -mcpu=mips32r2 -mattr=+micromips %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=mips -mcpu=mips32r6 -mattr=+micromips %s 2>&1 | FileCheck %s

        mfc0    $2, $config1, 0             # CHECK: :[[@LINE]]:29: error: explicit selector is not allowed for named COP0 register
        mtc0    $2, $status, 1              # CHECK: :[[@LINE]]:28: error: explicit selector is not allowed for named COP0 register
        mfc0    $2, $16, 8                  # CHECK: :[[@LINE]]:26: error: expected 3-bit unsigned immediate
        mtc0    $2, $16, -1                 # CHECK: :[[@LINE]]:26: error: expected 3-bit unsigned immediate
        mfc0    $2, $t0                     # CHECK: :[[@LINE]]:21: error: invalid operand for instruction
        addu    $config1, $2, $3            # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
        mfc0    $config1, $16, 1            # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
        mfc2    $2, $config1, 1             # CHECK: :[[@LINE]]:21: error: invalid operand for instruction
        mfc0    $2, $config1, 1             # CHECK: :[[@LINE]]:29: error: explicit selector is not allowed for named COP0 register
        mtc0    $2, $status, 0              # CHECK: :[[@LINE]]:28: error: explicit selector is not allowed for named COP0 register

        .set cp0name, $config1
        mfc0    $2, cp0name, 1              # CHECK: :[[@LINE]]:28: error: explicit selector is not allowed for named COP0 register
