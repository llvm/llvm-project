# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvvmm %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vmmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: vmmacc.vv v8, v4, v20, v0.t

vwmmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: vwmmacc.vv v8, v4, v20, v0.t

vqmmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: vqmmacc.vv v8, v4, v20, v0.t

v8wmmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: v8wmmacc.vv v8, v4, v20, v0.t
