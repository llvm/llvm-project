# RUN: not llvm-mc -triple=riscv32 --mattr=+experimental-zibi %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zibi %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
beqi a0, 0x0, 0x400
# CHECK-ERROR: [[@LINE-1]]:10: error: immediate must be non-zero in the range [-1, 31]
# CHECK-ERROR-LABEL: beqi a0, 0x0, 0x400
beqi a0, 0x21, 0x400
# CHECK-ERROR: [[@LINE-1]]:10: error: immediate must be non-zero in the range [-1, 31]
# CHECK-ERROR-LABEL: beqi a0, 0x21, 0x400
beqi a2, 0x10, -0x1f000
# CHECK-ERROR: [[@LINE-1]]:16: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
# CHECK-ERROR-LABEL: beqi a2, 0x10, -0x1f000
beqi a2, 0x10, 0x1000
# CHECK-ERROR: [[@LINE-1]]:16: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
# CHECK-ERROR-LABEL: beqi a2, 0x10, 0x1000
beqi a2, 0x10, 0x111
# CHECK-ERROR: [[@LINE-1]]:16: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
# CHECK-ERROR-LABEL: beqi a2, 0x10, 0x111
bnei a0, 0x0, 0x400
# CHECK-ERROR: [[@LINE-1]]:10: error: immediate must be non-zero in the range [-1, 31]
# CHECK-ERROR-LABEL: bnei a0, 0x0, 0x400
bnei a0, 0x21, 0x400
# CHECK-ERROR: [[@LINE-1]]:10: error: immediate must be non-zero in the range [-1, 31]
# CHECK-ERROR-LABEL: bnei a0, 0x21, 0x400
bnei a2, 0x10, -0x1f000
# CHECK-ERROR: [[@LINE-1]]:16: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
# CHECK-ERROR-LABEL: bnei a2, 0x10, -0x1f000
bnei a2, 0x10, 0x1000
# CHECK-ERROR: [[@LINE-1]]:16: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
# CHECK-ERROR-LABEL: bnei a2, 0x10, 0x1000
bnei a2, 0x10, 0x111
# CHECK-ERROR: [[@LINE-1]]:16: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
# CHECK-ERROR-LABEL: bnei a2, 0x10, 0x111
