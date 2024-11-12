// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fprcvt < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fprcvt < %s \
// RUN:        | llvm-objdump -d --mattr=+fprcvt - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fprcvt < %s \
// RUN:        | llvm-objdump -d  --no-print-imm-hex --mattr=-fprcvt - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fprcvt < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+fprcvt -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fcvtas s0, d1
// CHECK-INST: fcvtas s0, d1
// CHECK-ENCODING: [0x20,0x00,0x7a,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e7a0020 <unknown>

fcvtas s1, h2
// CHECK-INST: fcvtas s1, h2
// CHECK-ENCODING: [0x41,0x00,0xfa,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1efa0041 <unknown>

fcvtas d3, h4
// CHECK-INST: fcvtas d3, h4
// CHECK-ENCODING: [0x83,0x00,0xfa,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9efa0083 <unknown>

fcvtas d0, s5
// CHECK-INST: fcvtas d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x3a,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3a00a0 <unknown>

fcvtau s0, d1
// CHECK-INST: fcvtau s0, d1
// CHECK-ENCODING: [0x20,0x00,0x7b,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e7b0020 <unknown>

fcvtau s1, h2
// CHECK-INST: fcvtau s1, h2
// CHECK-ENCODING: [0x41,0x00,0xfb,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1efb0041 <unknown>

fcvtau d3, h4
// CHECK-INST: fcvtau d3, h4
// CHECK-ENCODING: [0x83,0x00,0xfb,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9efb0083 <unknown>

fcvtau d0, s5
// CHECK-INST: fcvtau d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x3b,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3b00a0 <unknown>

fcvtms s0, d1
// CHECK-INST: fcvtms s0, d1
// CHECK-ENCODING: [0x20,0x00,0x74,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e740020 <unknown>

fcvtms s1, h2
// CHECK-INST: fcvtms s1, h2
// CHECK-ENCODING: [0x41,0x00,0xf4,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1ef40041 <unknown>

fcvtms d3, h4
// CHECK-INST: fcvtms d3, h4
// CHECK-ENCODING: [0x83,0x00,0xf4,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9ef40083 <unknown>

fcvtms d0, s5
// CHECK-INST: fcvtms d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x34,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3400a0 <unknown>

fcvtmu s0, d1
// CHECK-INST: fcvtmu s0, d1
// CHECK-ENCODING: [0x20,0x00,0x75,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e750020 <unknown>

fcvtmu s1, h2
// CHECK-INST: fcvtmu s1, h2
// CHECK-ENCODING: [0x41,0x00,0xf5,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1ef50041 <unknown>

fcvtmu d3, h4
// CHECK-INST: fcvtmu d3, h4
// CHECK-ENCODING: [0x83,0x00,0xf5,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9ef50083 <unknown>

fcvtmu d0, s5
// CHECK-INST: fcvtmu d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x35,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3500a0 <unknown>

fcvtns s0, d1
// CHECK-INST: fcvtns s0, d1
// CHECK-ENCODING: [0x20,0x00,0x6a,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e6a0020 <unknown>

fcvtns s1, h2
// CHECK-INST: fcvtns s1, h2
// CHECK-ENCODING: [0x41,0x00,0xea,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1eea0041 <unknown>

fcvtns d3, h4
// CHECK-INST: fcvtns d3, h4
// CHECK-ENCODING: [0x83,0x00,0xea,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9eea0083 <unknown>

fcvtns d0, s5
// CHECK-INST: fcvtns d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x2a,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e2a00a0 <unknown>

fcvtnu s0, d1
// CHECK-INST: fcvtnu s0, d1
// CHECK-ENCODING: [0x20,0x00,0x6b,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e6b0020 <unknown>

fcvtnu s1, h2
// CHECK-INST: fcvtnu s1, h2
// CHECK-ENCODING: [0x41,0x00,0xeb,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1eeb0041 <unknown>

fcvtnu d3, h4
// CHECK-INST: fcvtnu d3, h4
// CHECK-ENCODING: [0x83,0x00,0xeb,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9eeb0083 <unknown>

fcvtnu d0, s5
// CHECK-INST: fcvtnu d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x2b,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e2b00a0 <unknown>

fcvtps s0, d1
// CHECK-INST: fcvtps s0, d1
// CHECK-ENCODING: [0x20,0x00,0x72,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e720020 <unknown>

fcvtps s1, h2
// CHECK-INST: fcvtps s1, h2
// CHECK-ENCODING: [0x41,0x00,0xf2,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1ef20041 <unknown>

fcvtps d3, h4
// CHECK-INST: fcvtps d3, h4
// CHECK-ENCODING: [0x83,0x00,0xf2,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9ef20083 <unknown>

fcvtps d0, s5
// CHECK-INST: fcvtps d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x32,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3200a0 <unknown>

fcvtpu s0, d1
// CHECK-INST: fcvtpu s0, d1
// CHECK-ENCODING: [0x20,0x00,0x73,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e730020 <unknown>

fcvtpu s1, h2
// CHECK-INST: fcvtpu s1, h2
// CHECK-ENCODING: [0x41,0x00,0xf3,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1ef30041 <unknown>

fcvtpu d3, h4
// CHECK-INST: fcvtpu d3, h4
// CHECK-ENCODING: [0x83,0x00,0xf3,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9ef30083 <unknown>

fcvtpu d0, s5
// CHECK-INST: fcvtpu d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x33,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3300a0 <unknown>

fcvtzs s0, d1
// CHECK-INST: fcvtzs s0, d1
// CHECK-ENCODING: [0x20,0x00,0x76,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e760020 <unknown>

fcvtzs s1, h2
// CHECK-INST: fcvtzs s1, h2
// CHECK-ENCODING: [0x41,0x00,0xf6,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1ef60041 <unknown>

fcvtzs d3, h4
// CHECK-INST: fcvtzs d3, h4
// CHECK-ENCODING: [0x83,0x00,0xf6,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9ef60083 <unknown>

fcvtzs d0, s5
// CHECK-INST: fcvtzs d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x36,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3600a0 <unknown>

fcvtzu s0, d1
// CHECK-INST: fcvtzu s0, d1
// CHECK-ENCODING: [0x20,0x00,0x77,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e770020 <unknown>

fcvtzu s1, h2
// CHECK-INST: fcvtzu s1, h2
// CHECK-ENCODING: [0x41,0x00,0xf7,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1ef70041 <unknown>

fcvtzu d3, h4
// CHECK-INST: fcvtzu d3, h4
// CHECK-ENCODING: [0x83,0x00,0xf7,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9ef70083 <unknown>

fcvtzu d0, s5
// CHECK-INST: fcvtzu d0, s5
// CHECK-ENCODING: [0xa0,0x00,0x37,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3700a0 <unknown>