// RUN: llvm-mc -triple aarch64 -show-encoding %s   | FileCheck %s

mrs x3, SCTLRMASK_EL1
// CHECK: mrs	x3, SCTLRMASK_EL1               // encoding: [0x03,0x14,0x38,0xd5]
mrs x3, SCTLRMASK_EL2
// CHECK: mrs	x3, SCTLRMASK_EL2               // encoding: [0x03,0x14,0x3c,0xd5]
mrs x3, SCTLRMASK_EL12
// CHECK: mrs	x3, SCTLRMASK_EL12              // encoding: [0x03,0x14,0x3d,0xd5]
mrs x3, CPACRMASK_EL1
// CHECK: mrs	x3, CPACRMASK_EL1               // encoding: [0x43,0x14,0x38,0xd5]
mrs x3, CPTRMASK_EL2
// CHECK: mrs	x3, CPTRMASK_EL2                // encoding: [0x43,0x14,0x3c,0xd5]
mrs x3, CPACRMASK_EL12
// CHECK: mrs	x3, CPACRMASK_EL12              // encoding: [0x43,0x14,0x3d,0xd5]
mrs x3, SCTLR2MASK_EL1
// CHECK: mrs	x3, SCTLR2MASK_EL1              // encoding: [0x63,0x14,0x38,0xd5]
mrs x3, SCTLR2MASK_EL2
// CHECK: mrs	x3, SCTLR2MASK_EL2              // encoding: [0x63,0x14,0x3c,0xd5]
mrs x3, SCTLR2MASK_EL12
// CHECK: mrs	x3, SCTLR2MASK_EL12             // encoding: [0x63,0x14,0x3d,0xd5]
mrs x3, CPACRALIAS_EL1
// CHECK: mrs	x3, CPACRALIAS_EL1              // encoding: [0x83,0x14,0x38,0xd5]
mrs x3, SCTLRALIAS_EL1
// CHECK: mrs	x3, SCTLRALIAS_EL1              // encoding: [0xc3,0x14,0x38,0xd5]
mrs x3, SCTLR2ALIAS_EL1
// CHECK: mrs	x3, SCTLR2ALIAS_EL1             // encoding: [0xe3,0x14,0x38,0xd5]
mrs x3, TCRMASK_EL1
// CHECK: mrs	x3, TCRMASK_EL1                 // encoding: [0x43,0x27,0x38,0xd5]
mrs x3, TCRMASK_EL2
// CHECK: mrs	x3, TCRMASK_EL2                 // encoding: [0x43,0x27,0x3c,0xd5]
mrs x3, TCRMASK_EL12
// CHECK: mrs	x3, TCRMASK_EL12                // encoding: [0x43,0x27,0x3d,0xd5]
mrs x3, TCR2MASK_EL1
// CHECK: mrs	x3, TCR2MASK_EL1                // encoding: [0x63,0x27,0x38,0xd5]
mrs x3, TCR2MASK_EL2
// CHECK: mrs	x3, TCR2MASK_EL2                // encoding: [0x63,0x27,0x3c,0xd5]
mrs x3, TCR2MASK_EL12
// CHECK: mrs	x3, TCR2MASK_EL12               // encoding: [0x63,0x27,0x3d,0xd5]
mrs x3, TCRALIAS_EL1
// CHECK: mrs	x3, TCRALIAS_EL1                // encoding: [0xc3,0x27,0x38,0xd5]
mrs x3, TCR2ALIAS_EL1
// CHECK: mrs	x3, TCR2ALIAS_EL1               // encoding: [0xe3,0x27,0x38,0xd5]
mrs x3, ACTLRMASK_EL1
// CHECK: mrs	x3, ACTLRMASK_EL1               // encoding: [0x23,0x14,0x38,0xd5]
mrs x3, ACTLRMASK_EL2
// CHECK: mrs	x3, ACTLRMASK_EL2               // encoding: [0x23,0x14,0x3c,0xd5]
mrs x3, ACTLRMASK_EL12
// CHECK: mrs	x3, ACTLRMASK_EL12              // encoding: [0x23,0x14,0x3d,0xd5]
mrs x3, ACTLRALIAS_EL1
// CHECK: mrs	x3, ACTLRALIAS_EL1              // encoding: [0xa3,0x14,0x38,0xd5]

msr SCTLRMASK_EL1, x3
// CHECK: msr	SCTLRMASK_EL1, x3               // encoding: [0x03,0x14,0x18,0xd5]
msr SCTLRMASK_EL2, x3
// CHECK: msr	SCTLRMASK_EL2, x3               // encoding: [0x03,0x14,0x1c,0xd5]
msr SCTLRMASK_EL12, x3
// CHECK: msr	SCTLRMASK_EL12, x3              // encoding: [0x03,0x14,0x1d,0xd5]
msr CPACRMASK_EL1, x3
// CHECK: msr	CPACRMASK_EL1, x3               // encoding: [0x43,0x14,0x18,0xd5]
msr CPTRMASK_EL2, x3
// CHECK: msr	CPTRMASK_EL2, x3                // encoding: [0x43,0x14,0x1c,0xd5]
msr CPACRMASK_EL12, x3
// CHECK: msr	CPACRMASK_EL12, x3              // encoding: [0x43,0x14,0x1d,0xd5]
msr SCTLR2MASK_EL1, x3
// CHECK: msr	SCTLR2MASK_EL1, x3              // encoding: [0x63,0x14,0x18,0xd5]
msr SCTLR2MASK_EL2, x3
// CHECK: msr	SCTLR2MASK_EL2, x3              // encoding: [0x63,0x14,0x1c,0xd5]
msr SCTLR2MASK_EL12, x3
// CHECK: msr	SCTLR2MASK_EL12, x3             // encoding: [0x63,0x14,0x1d,0xd5]
msr CPACRALIAS_EL1, x3
// CHECK: msr	CPACRALIAS_EL1, x3              // encoding: [0x83,0x14,0x18,0xd5]
msr SCTLRALIAS_EL1, x3
// CHECK: msr	SCTLRALIAS_EL1, x3              // encoding: [0xc3,0x14,0x18,0xd5]
msr SCTLR2ALIAS_EL1, x3
// CHECK: msr	SCTLR2ALIAS_EL1, x3             // encoding: [0xe3,0x14,0x18,0xd5]
msr TCRMASK_EL1, x3
// CHECK: msr	TCRMASK_EL1, x3                 // encoding: [0x43,0x27,0x18,0xd5]
msr TCRMASK_EL2, x3
// CHECK: msr	TCRMASK_EL2, x3                 // encoding: [0x43,0x27,0x1c,0xd5]
msr TCRMASK_EL12, x3
// CHECK: msr	TCRMASK_EL12, x3                // encoding: [0x43,0x27,0x1d,0xd5]
msr TCR2MASK_EL1, x3
// CHECK: msr	TCR2MASK_EL1, x3                // encoding: [0x63,0x27,0x18,0xd5]
msr TCR2MASK_EL2, x3
// CHECK: msr	TCR2MASK_EL2, x3                // encoding: [0x63,0x27,0x1c,0xd5]
msr TCR2MASK_EL12, x3
// CHECK: msr	TCR2MASK_EL12, x3               // encoding: [0x63,0x27,0x1d,0xd5]
msr TCRALIAS_EL1, x3
// CHECK: msr	TCRALIAS_EL1, x3                // encoding: [0xc3,0x27,0x18,0xd5]
msr TCR2ALIAS_EL1, x3
// CHECK: msr	TCR2ALIAS_EL1, x3               // encoding: [0xe3,0x27,0x18,0xd5]
msr ACTLRMASK_EL1, x3
// CHECK: msr	ACTLRMASK_EL1, x3               // encoding: [0x23,0x14,0x18,0xd5]
msr ACTLRMASK_EL2, x3
// CHECK: msr	ACTLRMASK_EL2, x3               // encoding: [0x23,0x14,0x1c,0xd5]
msr ACTLRMASK_EL12, x3
// CHECK: msr	ACTLRMASK_EL12, x3              // encoding: [0x23,0x14,0x1d,0xd5]
msr ACTLRALIAS_EL1, x3
// CHECK: msr	ACTLRALIAS_EL1, x3              // encoding: [0xa3,0x14,0x18,0xd5]



