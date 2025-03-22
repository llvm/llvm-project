# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xsfsclic < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

# RUN: not llvm-mc -triple riscv64 -mattr=-experimental-xsfsclic < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

csrrs t1, stvt, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'stvt' requires 'experimental-xsfsclic' to be enabled

csrrs t1, snxti, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'snxti' requires 'experimental-xsfsclic' to be enabled

csrrs t1, sintstatus, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sintstatus' requires 'experimental-xsfsclic' to be enabled

csrrs t1, sscratchcsw, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sscratchcsw' requires 'experimental-xsfsclic' to be enabled

csrrs t1, sscratchcswl, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sscratchcswl' requires 'experimental-xsfsclic' to be enabled
