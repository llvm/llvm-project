// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+tlbiw,+xs < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+tlbiw,+xs < %s \
// RUN:        | llvm-objdump -d --mattr=+tlbiw,+xs --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+tlbiw,+xs < %s \
// RUN:   | llvm-objdump -d --mattr=-tlbiw,-xs --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+tlbiw,+xs < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+tlbiw,+xs -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


tlbi VMALLWS2E1
// CHECK-INST: tlbi vmallws2e1
// CHECK-ENCODING: encoding: [0x5f,0x86,0x0c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:6: error: TLBI VMALLWS2E1 requires: tlbiw
// CHECK-UNKNOWN:  d50c865f      sys #4, c8, c6, #2

tlbi VMALLWS2E1IS
// CHECK-INST: tlbi vmallws2e1is
// CHECK-ENCODING: encoding: [0x5f,0x82,0x0c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:6: error: TLBI VMALLWS2E1IS requires: tlbiw
// CHECK-UNKNOWN:  d50c825f      sys #4, c8, c2, #2

tlbi VMALLWS2E1OS
// CHECK-INST: tlbi vmallws2e1os
// CHECK-ENCODING: encoding: [0x5f,0x85,0x0c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:6: error: TLBI VMALLWS2E1OS requires: tlbiw
// CHECK-UNKNOWN:  d50c855f      sys #4, c8, c5, #2

tlbi VMALLWS2E1nXS
// CHECK-INST: tlbi vmallws2e1nxs
// CHECK-ENCODING: encoding: [0x5f,0x96,0x0c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:6: error: TLBI VMALLWS2E1nXS requires: xs, tlbiw
// CHECK-UNKNOWN:  d50c965f      sys #4, c9, c6, #2

tlbi VMALLWS2E1ISnXS
// CHECK-INST: tlbi vmallws2e1isnxs
// CHECK-ENCODING: encoding: [0x5f,0x92,0x0c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:6: error: TLBI VMALLWS2E1ISnXS requires: xs, tlbiw
// CHECK-UNKNOWN:  d50c925f      sys #4, c9, c2, #2

tlbi VMALLWS2E1OSnXS
// CHECK-INST: tlbi vmallws2e1osnxs
// CHECK-ENCODING: encoding: [0x5f,0x95,0x0c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:6: error: TLBI VMALLWS2E1OSnXS requires: xs, tlbiw
// CHECK-UNKNOWN:  d50c955f      sys #4, c9, c5, #2
