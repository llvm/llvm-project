# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xsfmclic < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

# RUN: not llvm-mc -triple riscv64 -mattr=-experimental-xsfmclic < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

csrrs t1, sf.mtvt, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.mtvt' requires 'experimental-xsfmclic' to be enabled

csrrs t1, sf.mnxti, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.mnxti' requires 'experimental-xsfmclic' to be enabled

csrrs t1, sf.mintstatus, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.mintstatus' requires 'experimental-xsfmclic' to be enabled

csrrs t1, sf.mscratchcsw, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.mscratchcsw' requires 'experimental-xsfmclic' to be enabled

csrrs t1, sf.mscratchcswl, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.mscratchcswl' requires 'experimental-xsfmclic' to be enabled
