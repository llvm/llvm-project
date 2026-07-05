// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+vfp2,+mve,+8msecext -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK %s
// RUN:   FileCheck --check-prefix=ERROR < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=-vfp2,+mve,+8msecext -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-NOVFP %s
// RUN:   FileCheck --check-prefix=ERROR-NOVFP < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+vfp2,-mve,+8msecext -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-NOMVE %s
// RUN:   FileCheck --check-prefix=ERROR-NOMVE < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+vfp2,+mve,-8msecext -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-NOSEC %s
// RUN:   FileCheck --check-prefix=ERROR-NOSEC < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=-vfp2,-mve,-8msecext -show-encoding < %s 2>%t
// RUN:   FileCheck --check-prefix=ERROR-NONE < %t %s
// RUN: not llvm-mc -triple=thumbv8m.main-none-eabi -mattr=+vfp2,+8msecext -show-encoding < %s 2>%t
// RUN:   FileCheck --check-prefix=ERROR-V8M < %t %s

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vstr fpscr, [r0] @ encoding: [0x80,0xed,0x80,0x2f]
// CHECK-NOMVE: vstr fpscr, [r0] @ encoding: [0x80,0xed,0x80,0x2f]
// CHECK-NOVFP: vstr fpscr, [r0] @ encoding: [0x80,0xed,0x80,0x2f]
// CHECK: vstr fpscr, [r0] @ encoding: [0x80,0xed,0x80,0x2f]
vstr fpscr, [r0]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vstr fpscr_nzcvqc, [r9, #-24] @ encoding: [0x09,0xed,0x86,0x4f]
// CHECK-NOMVE: vstr fpscr_nzcvqc, [r9, #-24] @ encoding: [0x09,0xed,0x86,0x4f]
// CHECK-NOVFP: vstr fpscr_nzcvqc, [r9, #-24] @ encoding: [0x09,0xed,0x86,0x4f]
// CHECK: vstr fpscr_nzcvqc, [r9, #-24] @ encoding: [0x09,0xed,0x86,0x4f]
vstr fpscr_nzcvqc, [r9, #-24]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vstr fpscr_nzcvqc, [r9, #-24]! @ encoding: [0x29,0xed,0x86,0x4f]
// CHECK-NOMVE: vstr fpscr_nzcvqc, [r9, #-24]! @ encoding: [0x29,0xed,0x86,0x4f]
// CHECK-NOVFP: vstr fpscr_nzcvqc, [r9, #-24]! @ encoding: [0x29,0xed,0x86,0x4f]
// CHECK: vstr fpscr_nzcvqc, [r9, #-24]! @ encoding: [0x29,0xed,0x86,0x4f]
vstr fpscr_nzcvqc, [r9, #-24]!

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vstr fpscr_nzcvqc, [r9], #-24 @ encoding: [0x29,0xec,0x86,0x4f]
// CHECK-NOMVE: vstr fpscr_nzcvqc, [r9], #-24 @ encoding: [0x29,0xec,0x86,0x4f]
// CHECK-NOVFP: vstr fpscr_nzcvqc, [r9], #-24 @ encoding: [0x29,0xec,0x86,0x4f]
// CHECK: vstr fpscr_nzcvqc, [r9], #-24 @ encoding: [0x29,0xec,0x86,0x4f]
vstr fpscr_nzcvqc, [r9], #-24

// CHECK-NOSEC: it hi @ encoding: [0x88,0xbf]
// CHECK-NOMVE: it hi @ encoding: [0x88,0xbf]
// CHECK-NOVFP: it hi @ encoding: [0x88,0xbf]
// CHECK: it hi @ encoding: [0x88,0xbf]
it hi

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vstrhi fpscr, [r0] @ encoding: [0x80,0xed,0x80,0x2f]
// CHECK-NOMVE: vstrhi fpscr, [r0] @ encoding: [0x80,0xed,0x80,0x2f]
// CHECK-NOVFP: vstrhi fpscr, [r0] @ encoding: [0x80,0xed,0x80,0x2f]
// CHECK: vstrhi fpscr, [r0] @ encoding: [0x80,0xed,0x80,0x2f]
vstrhi fpscr, [r0]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vldr fpscr, [r0] @ encoding: [0x90,0xed,0x80,0x2f]
// CHECK-NOMVE: vldr fpscr, [r0] @ encoding: [0x90,0xed,0x80,0x2f]
// CHECK-NOVFP: vldr fpscr, [r0] @ encoding: [0x90,0xed,0x80,0x2f]
// CHECK: vldr fpscr, [r0] @ encoding: [0x90,0xed,0x80,0x2f]
vldr fpscr, [r0]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vldr fpscr_nzcvqc, [r9, #-24] @ encoding: [0x19,0xed,0x86,0x4f]
// CHECK-NOMVE: vldr fpscr_nzcvqc, [r9, #-24] @ encoding: [0x19,0xed,0x86,0x4f]
// CHECK-NOVFP: vldr fpscr_nzcvqc, [r9, #-24] @ encoding: [0x19,0xed,0x86,0x4f]
// CHECK: vldr fpscr_nzcvqc, [r9, #-24] @ encoding: [0x19,0xed,0x86,0x4f]
vldr fpscr_nzcvqc, [r9, #-24]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vldr fpscr_nzcvqc, [r9, #-24]! @ encoding: [0x39,0xed,0x86,0x4f]
// CHECK-NOMVE: vldr fpscr_nzcvqc, [r9, #-24]! @ encoding: [0x39,0xed,0x86,0x4f]
// CHECK-NOVFP: vldr fpscr_nzcvqc, [r9, #-24]! @ encoding: [0x39,0xed,0x86,0x4f]
// CHECK: vldr fpscr_nzcvqc, [r9, #-24]! @ encoding: [0x39,0xed,0x86,0x4f]
vldr fpscr_nzcvqc, [r9, #-24]!

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vldr fpscr_nzcvqc, [r9], #-24 @ encoding: [0x39,0xec,0x86,0x4f]
// CHECK-NOMVE: vldr fpscr_nzcvqc, [r9], #-24 @ encoding: [0x39,0xec,0x86,0x4f]
// CHECK-NOVFP: vldr fpscr_nzcvqc, [r9], #-24 @ encoding: [0x39,0xec,0x86,0x4f]
// CHECK: vldr fpscr_nzcvqc, [r9], #-24 @ encoding: [0x39,0xec,0x86,0x4f]
vldr fpscr_nzcvqc, [r9], #-24

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vldr fpscr_nzcvqc, [sp], #-52 @ encoding: [0x3d,0xec,0x8d,0x4f]
// CHECK-NOMVE: vldr fpscr_nzcvqc, [sp], #-52 @ encoding: [0x3d,0xec,0x8d,0x4f]
// CHECK-NOVFP: vldr fpscr_nzcvqc, [sp], #-52 @ encoding: [0x3d,0xec,0x8d,0x4f]
// CHECK: vldr fpscr_nzcvqc, [sp], #-52 @ encoding: [0x3d,0xec,0x8d,0x4f]
vldr fpscr_nzcvqc, [sp], #-52

// CHECK-NOSEC: it hi @ encoding: [0x88,0xbf]
// CHECK-NOMVE: it hi @ encoding: [0x88,0xbf]
// CHECK-NOVFP: it hi @ encoding: [0x88,0xbf]
// CHECK: it hi @ encoding: [0x88,0xbf]
it hi

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOSEC: vldrhi fpscr, [r0] @ encoding: [0x90,0xed,0x80,0x2f]
// CHECK-NOMVE: vldrhi fpscr, [r0] @ encoding: [0x90,0xed,0x80,0x2f]
// CHECK-NOVFP: vldrhi fpscr, [r0] @ encoding: [0x90,0xed,0x80,0x2f]
// CHECK: vldrhi fpscr, [r0] @ encoding: [0x90,0xed,0x80,0x2f]
vldrhi fpscr, [r0]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxts, [r12, #508] @ encoding: [0xcc,0xed,0xff,0xef]
// CHECK-NOVFP: vstr fpcxts, [r12, #508] @ encoding: [0xcc,0xed,0xff,0xef]
// CHECK: vstr fpcxts, [r12, #508] @ encoding: [0xcc,0xed,0xff,0xef]
vstr fpcxts, [r12, #508]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxts, [r12, #508]! @ encoding: [0xec,0xed,0xff,0xef]
// CHECK-NOVFP: vstr fpcxts, [r12, #508]! @ encoding: [0xec,0xed,0xff,0xef]
// CHECK: vstr fpcxts, [r12, #508]! @ encoding: [0xec,0xed,0xff,0xef]
vstr fpcxts, [r12, #508]!

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxts, [r12], #508 @ encoding: [0xec,0xec,0xff,0xef]
// CHECK-NOVFP: vstr fpcxts, [r12], #508 @ encoding: [0xec,0xec,0xff,0xef]
// CHECK: vstr fpcxts, [r12], #508 @ encoding: [0xec,0xec,0xff,0xef]
vstr fpcxts, [r12], #508

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxts, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xef]
// CHECK-NOVFP: vstr fpcxts, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xef]
// CHECK: vstr fpcxts, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xef]
vstr fpcxts, [sp], #-24

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxts, [r12, #508] @ encoding: [0xdc,0xed,0xff,0xef]
// CHECK-NOVFP: vldr fpcxts, [r12, #508] @ encoding: [0xdc,0xed,0xff,0xef]
// CHECK: vldr fpcxts, [r12, #508] @ encoding: [0xdc,0xed,0xff,0xef]
vldr fpcxts, [r12, #508]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxts, [r12, #508]! @ encoding: [0xfc,0xed,0xff,0xef]
// CHECK-NOVFP: vldr fpcxts, [r12, #508]! @ encoding: [0xfc,0xed,0xff,0xef]
// CHECK: vldr fpcxts, [r12, #508]! @ encoding: [0xfc,0xed,0xff,0xef]
vldr fpcxts, [r12, #508]!

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxts, [r12], #508 @ encoding: [0xfc,0xec,0xff,0xef]
// CHECK-NOVFP: vldr fpcxts, [r12], #508 @ encoding: [0xfc,0xec,0xff,0xef]
// CHECK: vldr fpcxts, [r12], #508 @ encoding: [0xfc,0xec,0xff,0xef]
vldr fpcxts, [r12], #508

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxts, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xef]
// CHECK-NOVFP: vldr fpcxts, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xef]
// CHECK: vldr fpcxts, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xef]
vldr fpcxts, [sp], #-24

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxtns, [r0] @ encoding: [0xc0,0xed,0x80,0xcf]
// CHECK-NOVFP: vstr fpcxtns, [r0] @ encoding: [0xc0,0xed,0x80,0xcf]
// CHECK: vstr fpcxtns, [r0] @ encoding: [0xc0,0xed,0x80,0xcf]
vstr fpcxtns, [r0]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxtns, [r9, #-24] @ encoding: [0x49,0xed,0x86,0xcf]
// CHECK-NOVFP: vstr fpcxtns, [r9, #-24] @ encoding: [0x49,0xed,0x86,0xcf]
// CHECK: vstr fpcxtns, [r9, #-24] @ encoding: [0x49,0xed,0x86,0xcf]
vstr fpcxtns, [r9, #-24]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxtns, [r6, #500] @ encoding: [0xc6,0xed,0xfd,0xcf]
// CHECK-NOVFP: vstr fpcxtns, [r6, #500] @ encoding: [0xc6,0xed,0xfd,0xcf]
// CHECK: vstr fpcxtns, [r6, #500] @ encoding: [0xc6,0xed,0xfd,0xcf]
vstr fpcxtns, [r6, #500]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxtns, [lr, #-508] @ encoding: [0x4e,0xed,0xff,0xcf]
// CHECK-NOVFP: vstr fpcxtns, [lr, #-508] @ encoding: [0x4e,0xed,0xff,0xcf]
// CHECK: vstr fpcxtns, [lr, #-508] @ encoding: [0x4e,0xed,0xff,0xcf]
vstr fpcxtns, [lr, #-508]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxtns, [r12, #508] @ encoding: [0xcc,0xed,0xff,0xcf]
// CHECK-NOVFP: vstr fpcxtns, [r12, #508] @ encoding: [0xcc,0xed,0xff,0xcf]
// CHECK: vstr fpcxtns, [r12, #508] @ encoding: [0xcc,0xed,0xff,0xcf]
vstr fpcxtns, [r12, #508]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vstr fpcxtns, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xcf]
// CHECK-NOVFP: vstr fpcxtns, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xcf]
// CHECK: vstr fpcxtns, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xcf]
vstr fpcxtns, [sp], #-24

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxtns, [r0] @ encoding: [0xd0,0xed,0x80,0xcf]
// CHECK-NOVFP: vldr fpcxtns, [r0] @ encoding: [0xd0,0xed,0x80,0xcf]
// CHECK: vldr fpcxtns, [r0] @ encoding: [0xd0,0xed,0x80,0xcf]
vldr fpcxtns, [r0]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxtns, [r9, #-24] @ encoding: [0x59,0xed,0x86,0xcf]
// CHECK-NOVFP: vldr fpcxtns, [r9, #-24] @ encoding: [0x59,0xed,0x86,0xcf]
// CHECK: vldr fpcxtns, [r9, #-24] @ encoding: [0x59,0xed,0x86,0xcf]
vldr fpcxtns, [r9, #-24]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxtns, [r6, #500] @ encoding: [0xd6,0xed,0xfd,0xcf]
// CHECK-NOVFP: vldr fpcxtns, [r6, #500] @ encoding: [0xd6,0xed,0xfd,0xcf]
// CHECK: vldr fpcxtns, [r6, #500] @ encoding: [0xd6,0xed,0xfd,0xcf]
vldr fpcxtns, [r6, #500]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxtns, [lr, #-508] @ encoding: [0x5e,0xed,0xff,0xcf]
// CHECK-NOVFP: vldr fpcxtns, [lr, #-508] @ encoding: [0x5e,0xed,0xff,0xcf]
// CHECK: vldr fpcxtns, [lr, #-508] @ encoding: [0x5e,0xed,0xff,0xcf]
vldr fpcxtns, [lr, #-508]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxtns, [r12, #508] @ encoding: [0xdc,0xed,0xff,0xcf]
// CHECK-NOVFP: vldr fpcxtns, [r12, #508] @ encoding: [0xdc,0xed,0xff,0xcf]
// CHECK: vldr fpcxtns, [r12, #508] @ encoding: [0xdc,0xed,0xff,0xcf]
vldr fpcxtns, [r12, #508]

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK-NOMVE: vldr fpcxtns, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xcf]
// CHECK-NOVFP: vldr fpcxtns, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xcf]
// CHECK: vldr fpcxtns, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xcf]
vldr fpcxtns, [sp], #-24

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vstr vpr, [r6, #500] @ encoding: [0xc6,0xed,0xfd,0x8f]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vstr vpr, [r6, #500] @ encoding: [0xc6,0xed,0xfd,0x8f]
// CHECK: vstr vpr, [r6, #500] @ encoding: [0xc6,0xed,0xfd,0x8f]
vstr vpr, [r6, #500]

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vstr p0, [lr, #-508] @ encoding: [0x4e,0xed,0xff,0xaf]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vstr p0, [lr, #-508] @ encoding: [0x4e,0xed,0xff,0xaf]
// CHECK: vstr p0, [lr, #-508] @ encoding: [0x4e,0xed,0xff,0xaf]
vstr p0, [lr, #-508]

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vstr vpr, [r6, #500]! @ encoding: [0xe6,0xed,0xfd,0x8f]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vstr vpr, [r6, #500]! @ encoding: [0xe6,0xed,0xfd,0x8f]
// CHECK: vstr vpr, [r6, #500]! @ encoding: [0xe6,0xed,0xfd,0x8f]
vstr vpr, [r6, #500]!

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vstr p0, [lr, #-508]! @ encoding: [0x6e,0xed,0xff,0xaf]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vstr p0, [lr, #-508]! @ encoding: [0x6e,0xed,0xff,0xaf]
// CHECK: vstr p0, [lr, #-508]! @ encoding: [0x6e,0xed,0xff,0xaf]
vstr p0, [lr, #-508]!

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vstr vpr, [r6], #500 @ encoding: [0xe6,0xec,0xfd,0x8f]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vstr vpr, [r6], #500 @ encoding: [0xe6,0xec,0xfd,0x8f]
// CHECK: vstr vpr, [r6], #500 @ encoding: [0xe6,0xec,0xfd,0x8f]
vstr vpr, [r6], #500

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vstr p0, [lr], #-508 @ encoding: [0x6e,0xec,0xff,0xaf]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vstr p0, [lr], #-508 @ encoding: [0x6e,0xec,0xff,0xaf]
// CHECK: vstr p0, [lr], #-508 @ encoding: [0x6e,0xec,0xff,0xaf]
vstr p0, [lr], #-508

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vstr p0, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xaf]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vstr p0, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xaf]
// CHECK: vstr p0, [sp], #-24 @ encoding: [0x6d,0xec,0x86,0xaf]
vstr p0, [sp], #-24

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vldr vpr, [r6, #500] @ encoding: [0xd6,0xed,0xfd,0x8f]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vldr vpr, [r6, #500] @ encoding: [0xd6,0xed,0xfd,0x8f]
// CHECK: vldr vpr, [r6, #500] @ encoding: [0xd6,0xed,0xfd,0x8f]
vldr vpr, [r6, #500]

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vldr p0, [lr, #-508] @ encoding: [0x5e,0xed,0xff,0xaf]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vldr p0, [lr, #-508] @ encoding: [0x5e,0xed,0xff,0xaf]
// CHECK: vldr p0, [lr, #-508] @ encoding: [0x5e,0xed,0xff,0xaf]
vldr p0, [lr, #-508]

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vldr vpr, [r6, #500]! @ encoding: [0xf6,0xed,0xfd,0x8f]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vldr vpr, [r6, #500]! @ encoding: [0xf6,0xed,0xfd,0x8f]
// CHECK: vldr vpr, [r6, #500]! @ encoding: [0xf6,0xed,0xfd,0x8f]
vldr vpr, [r6, #500]!

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vldr p0, [lr, #-508]! @ encoding: [0x7e,0xed,0xff,0xaf]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vldr p0, [lr, #-508]! @ encoding: [0x7e,0xed,0xff,0xaf]
// CHECK: vldr p0, [lr, #-508]! @ encoding: [0x7e,0xed,0xff,0xaf]
vldr p0, [lr, #-508]!

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vldr vpr, [r6], #500 @ encoding: [0xf6,0xec,0xfd,0x8f]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vldr vpr, [r6], #500 @ encoding: [0xf6,0xec,0xfd,0x8f]
// CHECK: vldr vpr, [r6], #500 @ encoding: [0xf6,0xec,0xfd,0x8f]
vldr vpr, [r6], #500

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vldr p0, [lr], #-508 @ encoding: [0x7e,0xec,0xff,0xaf]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vldr p0, [lr], #-508 @ encoding: [0x7e,0xec,0xff,0xaf]
// CHECK: vldr p0, [lr], #-508 @ encoding: [0x7e,0xec,0xff,0xaf]
vldr p0, [lr], #-508

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// CHECK-NOSEC: vldr p0, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xaf]
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOVFP: vldr p0, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xaf]
// CHECK: vldr p0, [sp], #-24 @ encoding: [0x7d,0xec,0x86,0xaf]
vldr p0, [sp], #-24

// ERROR-NOSEC: invalid instruction
// ERROR-NOMVE: invalid operand for instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr fpcxtns, [pc, #4]!

// ERROR-NOSEC: invalid instruction
// ERROR-NOMVE: invalid operand for instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vstr fpcxtns, [r0, #-507]

// ERROR-NOSEC: invalid instruction
// ERROR-NOMVE: invalid operand for instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr fpcxtns, [r2, #512]

// ERROR-NOSEC: invalid instruction
// ERROR-NOMVE: invalid operand for instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr fpcxtns, [pc], #-24

// ERROR-NOSEC: invalid operand for instruction
// ERROR-NOMVE: invalid operand for instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vstr vpr, [r0, #-507]

// ERROR-NOSEC: invalid operand for instruction
// ERROR-NOMVE: invalid instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr p0, [r2, #512]

// ERROR-NOSEC: invalid operand for instruction
// ERROR-NOMVE: invalid instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr p0, [r2, #2]

// ERROR-NOSEC: invalid operand for instruction
// ERROR-NOMVE: invalid instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr p0, [pc], #4

// ERROR-NOSEC: invalid operand for instruction
// ERROR-NOMVE: invalid instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr fpscr, [pc, #4]!

// ERROR-NOSEC: invalid operand for instruction
// ERROR-NOMVE: invalid operand for instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr fpscr_nzcvqc, [r8], #-53

// ERROR-NOSEC: invalid operand for instruction
// ERROR-NOMVE: invalid operand for instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr fpscr_nzcvqc, [r8], #2

// ERROR-NOSEC: invalid operand for instruction
// ERROR-NOMVE: invalid operand for instruction
// ERROR-NOVFP: invalid operand for instruction
// ERROR: invalid operand for instruction
vldr  fpscr_nzcvqc, [pc], #-52

