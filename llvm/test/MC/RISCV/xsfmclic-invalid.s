# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xsfmclic < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

# RUN: not llvm-mc -triple riscv64 -mattr=-experimental-xsfmclic < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

csrrs t1, mtvt, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'mtvt' requires 'experimental-xsfmclic' to be enabled

csrrs t1, mnxti, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'mnxti' requires 'experimental-xsfmclic' to be enabled

csrrs t1, mintstatus, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'mintstatus' requires 'experimental-xsfmclic' to be enabled

csrrs t1, mscratchcsw, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'mscratchcsw' requires 'experimental-xsfmclic' to be enabled

csrrs t1, mscratchcswl, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'mscratchcswl' requires 'experimental-xsfmclic' to be enabled
