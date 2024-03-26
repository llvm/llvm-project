// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+tlbiw -mattr=+xs < %s | FileCheck --check-prefix=CHECK-TLBIW --check-prefix=CHECK-XS %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+tlbiw < %s 2> %t | FileCheck --check-prefix=CHECK-TLBIW %s && FileCheck --check-prefix=ERROR-NO-XS-TLBIW %s < %t
// RUN: not llvm-mc -triple aarch64 < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-TLBIW --check-prefix=ERROR-NO-XS-TLBIW %s

tlbi VMALLWS2E1
// CHECK-TLBIW: tlbi vmallws2e1                  // encoding: [0x5f,0x86,0x0c,0xd5]
// ERROR-NO-TLBIW: [[@LINE-2]]:6: error: TLBI VMALLWS2E1 requires: tlbiw

tlbi VMALLWS2E1IS
// CHECK-TLBIW: tlbi vmallws2e1is                // encoding: [0x5f,0x82,0x0c,0xd5]
// ERROR-NO-TLBIW: [[@LINE-2]]:6: error: TLBI VMALLWS2E1IS requires: tlbiw

tlbi VMALLWS2E1OS
// CHECK-TLBIW: tlbi vmallws2e1os                // encoding: [0x5f,0x85,0x0c,0xd5]
// ERROR-NO-TLBIW: [[@LINE-2]]:6: error: TLBI VMALLWS2E1OS requires: tlbiw

tlbi VMALLWS2E1nXS
// CHECK-XS: tlbi vmallws2e1nxs                  // encoding: [0x5f,0x96,0x0c,0xd5]
// ERROR-NO-XS-TLBIW: [[@LINE-2]]:6: error: TLBI VMALLWS2E1nXS requires: xs, tlbiw

tlbi VMALLWS2E1ISnXS
// CHECK-XS: tlbi vmallws2e1isnxs                // encoding: [0x5f,0x92,0x0c,0xd5]
// ERROR-NO-XS-TLBIW: [[@LINE-2]]:6: error: TLBI VMALLWS2E1ISnXS requires: xs, tlbiw

tlbi VMALLWS2E1OSnXS
// CHECK-XS: tlbi vmallws2e1osnxs                // encoding: [0x5f,0x95,0x0c,0xd5]
// ERROR-NO-XS-TLBIW: [[@LINE-2]]:6: error: TLBI VMALLWS2E1OSnXS requires: xs, tlbiw
