// RUN: not llvm-mc -triple=armv7a-arm-none-eabi -mattr=+vfp2 -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-V7A-ARM %s
// RUN:   FileCheck --check-prefix=ERROR-V7A-ARM < %t %s
// RUN: not llvm-mc -triple=thumbv7a-arm-none-eabi -mattr=+vfp2 -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-V7A-THUMB %s
// RUN:   FileCheck --check-prefix=ERROR-V7A-THUMB < %t %s
// RUN: not llvm-mc -triple=thumbv7m-arm-none-eabi -mattr=+vfp2 -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-V7M %s
// RUN:   FileCheck --check-prefix=ERROR-V7M < %t %s
// RUN: not llvm-mc -triple=armv8a-arm-none-eabi -mattr=+fp-armv8 -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-V8A-ARM %s
// RUN:   FileCheck --check-prefix=ERROR-V8A-ARM < %t %s
// RUN: not llvm-mc -triple=thumbv8a-arm-none-eabi -mattr=+fp-armv8 -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-V8A-THUMB %s
// RUN:   FileCheck --check-prefix=ERROR-V8A-THUMB < %t %s
// RUN: not llvm-mc -triple=thumbv8m.main-arm-none-eabi -mattr=+fp-armv8 -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-V8M %s
// RUN:   FileCheck --check-prefix=ERROR-V8M < %t %s
// RUN: not llvm-mc -triple=thumbv7m-arm-none-eabi -show-encoding < %s 2>%t
// RUN:   FileCheck --check-prefix=ERROR-NOVFP < %t %s

        vmrs    APSR_nzcv, fpscr
        vmrs    apsr_nzcv, fpscr
        fmstat
        vmrs    r10, fpscr
        vmrs    r2, fpsid
        vmrs    r3, FPSID
        vmrs    r4, mvfr0
        vmrs    r5, MVFR1
        vmrs    r6, mvfr2
        vmrs    sp, fpscr
        vmrs    pc, fpscr

// CHECK-V7A-ARM: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
// CHECK-V7A-ARM: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
// CHECK-V7A-ARM: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
// CHECK-V7A-ARM: vmrs r10, fpscr             @ encoding: [0x10,0xaa,0xf1,0xee]
// CHECK-V7A-ARM: vmrs r2, fpsid              @ encoding: [0x10,0x2a,0xf0,0xee]
// CHECK-V7A-ARM: vmrs r3, fpsid              @ encoding: [0x10,0x3a,0xf0,0xee]
// CHECK-V7A-ARM: vmrs r4, mvfr0              @ encoding: [0x10,0x4a,0xf7,0xee]
// CHECK-V7A-ARM: vmrs r5, mvfr1              @ encoding: [0x10,0x5a,0xf6,0xee]
// ERROR-V7A-ARM: instruction requires: FPARMv8
// CHECK-V7A-ARM: vmrs sp, fpscr              @ encoding: [0x10,0xda,0xf1,0xee]
// ERROR-V7A-ARM: invalid operand for instruction

// CHECK-V7A-THUMB: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V7A-THUMB: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V7A-THUMB: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V7A-THUMB: vmrs r10, fpscr             @ encoding: [0xf1,0xee,0x10,0xaa]
// CHECK-V7A-THUMB: vmrs r2, fpsid              @ encoding: [0xf0,0xee,0x10,0x2a]
// CHECK-V7A-THUMB: vmrs r3, fpsid              @ encoding: [0xf0,0xee,0x10,0x3a]
// CHECK-V7A-THUMB: vmrs r4, mvfr0              @ encoding: [0xf7,0xee,0x10,0x4a]
// CHECK-V7A-THUMB: vmrs r5, mvfr1              @ encoding: [0xf6,0xee,0x10,0x5a]
// ERROR-V7A-THUMB: instruction requires: FPARMv8
// ERROR-V7A-THUMB: invalid operand for instruction
// ERROR-V7A-THUMB: invalid operand for instruction

// CHECK-V7M: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V7M: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V7M: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V7M: vmrs r10, fpscr             @ encoding: [0xf1,0xee,0x10,0xaa]
// CHECK-V7M: vmrs r2, fpsid              @ encoding: [0xf0,0xee,0x10,0x2a]
// CHECK-V7M: vmrs r3, fpsid              @ encoding: [0xf0,0xee,0x10,0x3a]
// CHECK-V7M: vmrs r4, mvfr0              @ encoding: [0xf7,0xee,0x10,0x4a]
// CHECK-V7M: vmrs r5, mvfr1              @ encoding: [0xf6,0xee,0x10,0x5a]
// ERROR-V7M: instruction requires: FPARMv8
// ERROR-V7M: invalid operand for instruction
// ERROR-V7M: invalid operand for instruction

// CHECK-V8A-ARM: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
// CHECK-V8A-ARM: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
// CHECK-V8A-ARM: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
// CHECK-V8A-ARM: vmrs r10, fpscr             @ encoding: [0x10,0xaa,0xf1,0xee]
// CHECK-V8A-ARM: vmrs r2, fpsid              @ encoding: [0x10,0x2a,0xf0,0xee]
// CHECK-V8A-ARM: vmrs r3, fpsid              @ encoding: [0x10,0x3a,0xf0,0xee]
// CHECK-V8A-ARM: vmrs r4, mvfr0              @ encoding: [0x10,0x4a,0xf7,0xee]
// CHECK-V8A-ARM: vmrs r5, mvfr1              @ encoding: [0x10,0x5a,0xf6,0xee]
// CHECK-V8A-ARM: vmrs r6, mvfr2              @ encoding: [0x10,0x6a,0xf5,0xee]
// CHECK-V8A-ARM: vmrs sp, fpscr              @ encoding: [0x10,0xda,0xf1,0xee]
// ERROR-V8A-ARM: invalid operand for instruction

// CHECK-V8A-THUMB: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V8A-THUMB: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V8A-THUMB: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V8A-THUMB: vmrs r10, fpscr             @ encoding: [0xf1,0xee,0x10,0xaa]
// CHECK-V8A-THUMB: vmrs r2, fpsid              @ encoding: [0xf0,0xee,0x10,0x2a]
// CHECK-V8A-THUMB: vmrs r3, fpsid              @ encoding: [0xf0,0xee,0x10,0x3a]
// CHECK-V8A-THUMB: vmrs r4, mvfr0              @ encoding: [0xf7,0xee,0x10,0x4a]
// CHECK-V8A-THUMB: vmrs r5, mvfr1              @ encoding: [0xf6,0xee,0x10,0x5a]
// CHECK-V8A-THUMB: vmrs r6, mvfr2              @ encoding: [0xf5,0xee,0x10,0x6a]
// CHECK-V8A-THUMB: vmrs sp, fpscr              @ encoding: [0xf1,0xee,0x10,0xda]
// ERROR-V8A-THUMB: invalid operand for instruction

// CHECK-V8M: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V8M: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V8M: vmrs APSR_nzcv, fpscr       @ encoding: [0xf1,0xee,0x10,0xfa]
// CHECK-V8M: vmrs r10, fpscr             @ encoding: [0xf1,0xee,0x10,0xaa]
// CHECK-V8M: vmrs r2, fpsid              @ encoding: [0xf0,0xee,0x10,0x2a]
// CHECK-V8M: vmrs r3, fpsid              @ encoding: [0xf0,0xee,0x10,0x3a]
// CHECK-V8M: vmrs r4, mvfr0              @ encoding: [0xf7,0xee,0x10,0x4a]
// CHECK-V8M: vmrs r5, mvfr1              @ encoding: [0xf6,0xee,0x10,0x5a]
// CHECK-V8M: vmrs r6, mvfr2              @ encoding: [0xf5,0xee,0x10,0x6a]
// ERROR-V8M: invalid operand for instruction
// ERROR-V8M: invalid operand for instruction

// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: FPARMv8
// ERROR-NOVFP: invalid instruction
// ERROR-NOVFP: invalid instruction

        vmsr  fpscr, APSR_nzcv
        vmsr  fpscr, r0
        vmsr  fpexc, r1
        vmsr  fpsid, r2
        vmsr  fpscr, r10
        vmsr  fpscr, sp
        vmsr  fpscr, pc

// ERROR-V7A-ARM: operand must be a register in range [r0, r14]
// CHECK-V7A-ARM: vmsr  fpscr, r0             @ encoding: [0x10,0x0a,0xe1,0xee]
// CHECK-V7A-ARM: vmsr  fpexc, r1             @ encoding: [0x10,0x1a,0xe8,0xee]
// CHECK-V7A-ARM: vmsr  fpsid, r2             @ encoding: [0x10,0x2a,0xe0,0xee]
// CHECK-V7A-ARM: vmsr  fpscr, r10            @ encoding: [0x10,0xaa,0xe1,0xee]
// CHECK-V7A-ARM: vmsr  fpscr, sp             @ encoding: [0x10,0xda,0xe1,0xee]
// ERROR-V7A-ARM: operand must be a register in range [r0, r14]

// ERROR-V7A-THUMB: operand must be a register in range [r0, r14]
// CHECK-V7A-THUMB: vmsr  fpscr, r0             @ encoding: [0xe1,0xee,0x10,0x0a]
// CHECK-V7A-THUMB: vmsr  fpexc, r1             @ encoding: [0xe8,0xee,0x10,0x1a]
// CHECK-V7A-THUMB: vmsr  fpsid, r2             @ encoding: [0xe0,0xee,0x10,0x2a]
// CHECK-V7A-THUMB: vmsr  fpscr, r10            @ encoding: [0xe1,0xee,0x10,0xaa]
// ERROR-V7A-THUMB: invalid operand for instruction
// ERROR-V7A-THUMB: operand must be a register in range [r0, r14]

// ERROR-V7M: operand must be a register in range [r0, r14]
// CHECK-V7M: vmsr  fpscr, r0             @ encoding: [0xe1,0xee,0x10,0x0a]
// CHECK-V7M: vmsr  fpexc, r1             @ encoding: [0xe8,0xee,0x10,0x1a]
// CHECK-V7M: vmsr  fpsid, r2             @ encoding: [0xe0,0xee,0x10,0x2a]
// CHECK-V7M: vmsr  fpscr, r10            @ encoding: [0xe1,0xee,0x10,0xaa]
// ERROR-V7M: invalid operand for instruction
// ERROR-V7M: operand must be a register in range [r0, r14]

// ERROR-V8A-ARM: operand must be a register in range [r0, r14]
// CHECK-V8A-ARM: vmsr  fpscr, r0             @ encoding: [0x10,0x0a,0xe1,0xee]
// CHECK-V8A-ARM: vmsr  fpexc, r1             @ encoding: [0x10,0x1a,0xe8,0xee]
// CHECK-V8A-ARM: vmsr  fpsid, r2             @ encoding: [0x10,0x2a,0xe0,0xee]
// CHECK-V8A-ARM: vmsr  fpscr, r10            @ encoding: [0x10,0xaa,0xe1,0xee]
// CHECK-V8A-ARM: vmsr  fpscr, sp             @ encoding: [0x10,0xda,0xe1,0xee]
// ERROR-V8A-ARM: operand must be a register in range [r0, r14]

// ERROR-V8A-THUMB: operand must be a register in range [r0, r14]
// CHECK-V8A-THUMB: vmsr  fpscr, r0             @ encoding: [0xe1,0xee,0x10,0x0a]
// CHECK-V8A-THUMB: vmsr  fpexc, r1             @ encoding: [0xe8,0xee,0x10,0x1a]
// CHECK-V8A-THUMB: vmsr  fpsid, r2             @ encoding: [0xe0,0xee,0x10,0x2a]
// CHECK-V8A-THUMB: vmsr  fpscr, r10            @ encoding: [0xe1,0xee,0x10,0xaa]
// CHECK-V8A-THUMB: vmsr  fpscr, sp             @ encoding: [0xe1,0xee,0x10,0xda]
// ERROR-V8A-THUMB: operand must be a register in range [r0, r14]

// ERROR-V8M: operand must be a register in range [r0, r14]
// CHECK-V8M: vmsr  fpscr, r0             @ encoding: [0xe1,0xee,0x10,0x0a]
// CHECK-V8M: vmsr  fpexc, r1             @ encoding: [0xe8,0xee,0x10,0x1a]
// CHECK-V8M: vmsr  fpsid, r2             @ encoding: [0xe0,0xee,0x10,0x2a]
// CHECK-V8M: vmsr  fpscr, r10            @ encoding: [0xe1,0xee,0x10,0xaa]
// ERROR-V8M: invalid operand for instruction
// ERROR-V8M: operand must be a register in range [r0, r14]

// ERROR-NOVFP: invalid instruction
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: instruction requires: VFP2
// ERROR-NOVFP: invalid instruction
// ERROR-NOVFP: invalid instruction
