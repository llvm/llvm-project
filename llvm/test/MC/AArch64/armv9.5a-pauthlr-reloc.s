// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+pauth-lr -filetype=obj -o /dev/null 2>&1 < %s | FileCheck %s

  autiasppc undef_label
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: relocation of PAC/AUT instructions is not supported
// CHECK-NEXT: autiasppc undef_label
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

  autibsppc undef_label
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: relocation of PAC/AUT instructions is not supported
// CHECK-NEXT: autibsppc undef_label
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

