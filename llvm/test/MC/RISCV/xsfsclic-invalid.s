# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xsfsclic < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

# RUN: not llvm-mc -triple riscv64 -mattr=-experimental-xsfsclic < %s 2>&1 \
# RUN:         | FileCheck -check-prefixes=CHECK-FEATURE %s

csrrs t1, sf.stvt, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.stvt' requires 'experimental-xsfsclic' to be enabled

csrrs t1, sf.snxti, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.snxti' requires 'experimental-xsfsclic' to be enabled

csrrs t1, sf.sintstatus, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.sintstatus' requires 'experimental-xsfsclic' to be enabled

csrrs t1, sf.sscratchcsw, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.sscratchcsw' requires 'experimental-xsfsclic' to be enabled

csrrs t1, sf.sscratchcswl, zero
// CHECK-FEATURE: :[[@LINE-1]]:11: error: system register 'sf.sscratchcswl' requires 'experimental-xsfsclic' to be enabled
