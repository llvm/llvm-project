// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding               -mattr=+ite < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.8a -mattr=+ite < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.9a -mattr=+ite < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.3a -mattr=+ite < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.4a -mattr=+ite < %s | FileCheck %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding               < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-ITE %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.8a < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-ITE %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.9a < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-ITE %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.3a < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-ITE %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.4a < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-ITE %s

// FEAT_DEBUGv8p9
            mrs	x3, MDSELR_EL1
// CHECK:   mrs	x3, MDSELR_EL1                  // encoding: [0x43,0x04,0x30,0xd5]
            msr MDSELR_EL1, x1
// CHECK:   msr	MDSELR_EL1, x1                  // encoding: [0x41,0x04,0x10,0xd5]

// FEAT_PMUv3p9
            mrs	x3, PMUACR_EL1
// CHECK:   mrs	x3, PMUACR_EL1                  // encoding: [0x83,0x9e,0x38,0xd5]
            msr	PMUACR_EL1, x1
// CHECK:   msr	PMUACR_EL1, x1                  // encoding: [0x81,0x9e,0x18,0xd5]

// FEAT_PMUv3_SS
            mrs	x3, PMCCNTSVR_EL1
// CHECK:   mrs	x3, PMCCNTSVR_EL1               // encoding: [0xe3,0xeb,0x30,0xd5]
            mrs	x3, PMICNTSVR_EL1
// CHECK:   mrs	x3, PMICNTSVR_EL1               // encoding: [0x03,0xec,0x30,0xd5]
            mrs	x3, PMSSCR_EL1
// CHECK:   mrs	x3, PMSSCR_EL1                  // encoding: [0x63,0x9d,0x38,0xd5]
            msr	PMSSCR_EL1, x1
// CHECK:   msr	PMSSCR_EL1, x1                  // encoding: [0x61,0x9d,0x18,0xd5]
            mrs	x3, PMEVCNTSVR0_EL1
// CHECK:   mrs	x3, PMEVCNTSVR0_EL1             // encoding: [0x03,0xe8,0x30,0xd5]
            mrs	x3, PMEVCNTSVR1_EL1
// CHECK:   mrs	x3, PMEVCNTSVR1_EL1             // encoding: [0x23,0xe8,0x30,0xd5]
            mrs	x3, PMEVCNTSVR2_EL1
// CHECK:   mrs	x3, PMEVCNTSVR2_EL1             // encoding: [0x43,0xe8,0x30,0xd5]
            mrs	x3, PMEVCNTSVR3_EL1
// CHECK:   mrs	x3, PMEVCNTSVR3_EL1             // encoding: [0x63,0xe8,0x30,0xd5]
            mrs	x3, PMEVCNTSVR4_EL1
// CHECK:   mrs	x3, PMEVCNTSVR4_EL1             // encoding: [0x83,0xe8,0x30,0xd5]
            mrs	x3, PMEVCNTSVR5_EL1
// CHECK:   mrs	x3, PMEVCNTSVR5_EL1             // encoding: [0xa3,0xe8,0x30,0xd5]
            mrs	x3, PMEVCNTSVR6_EL1
// CHECK:   mrs	x3, PMEVCNTSVR6_EL1             // encoding: [0xc3,0xe8,0x30,0xd5]
            mrs	x3, PMEVCNTSVR7_EL1
// CHECK:   mrs	x3, PMEVCNTSVR7_EL1             // encoding: [0xe3,0xe8,0x30,0xd5]
            mrs	x3, PMEVCNTSVR8_EL1
// CHECK:   mrs	x3, PMEVCNTSVR8_EL1             // encoding: [0x03,0xe9,0x30,0xd5]
            mrs	x3, PMEVCNTSVR9_EL1
// CHECK:   mrs	x3, PMEVCNTSVR9_EL1             // encoding: [0x23,0xe9,0x30,0xd5]
            mrs	x3, PMEVCNTSVR10_EL1
// CHECK:   mrs	x3, PMEVCNTSVR10_EL1            // encoding: [0x43,0xe9,0x30,0xd5]
            mrs	x3, PMEVCNTSVR11_EL1
// CHECK:   mrs	x3, PMEVCNTSVR11_EL1            // encoding: [0x63,0xe9,0x30,0xd5]
            mrs	x3, PMEVCNTSVR12_EL1
// CHECK:   mrs	x3, PMEVCNTSVR12_EL1            // encoding: [0x83,0xe9,0x30,0xd5]
            mrs	x3, PMEVCNTSVR13_EL1
// CHECK:   mrs	x3, PMEVCNTSVR13_EL1            // encoding: [0xa3,0xe9,0x30,0xd5]
            mrs	x3, PMEVCNTSVR14_EL1
// CHECK:   mrs	x3, PMEVCNTSVR14_EL1            // encoding: [0xc3,0xe9,0x30,0xd5]
            mrs	x3, PMEVCNTSVR15_EL1
// CHECK:   mrs	x3, PMEVCNTSVR15_EL1            // encoding: [0xe3,0xe9,0x30,0xd5]
            mrs	x3, PMEVCNTSVR16_EL1
// CHECK:   mrs	x3, PMEVCNTSVR16_EL1            // encoding: [0x03,0xea,0x30,0xd5]
            mrs	x3, PMEVCNTSVR17_EL1
// CHECK:   mrs	x3, PMEVCNTSVR17_EL1            // encoding: [0x23,0xea,0x30,0xd5]
            mrs	x3, PMEVCNTSVR18_EL1
// CHECK:   mrs	x3, PMEVCNTSVR18_EL1            // encoding: [0x43,0xea,0x30,0xd5]
            mrs	x3, PMEVCNTSVR19_EL1
// CHECK:   mrs	x3, PMEVCNTSVR19_EL1            // encoding: [0x63,0xea,0x30,0xd5]
            mrs	x3, PMEVCNTSVR20_EL1
// CHECK:   mrs	x3, PMEVCNTSVR20_EL1            // encoding: [0x83,0xea,0x30,0xd5]
            mrs	x3, PMEVCNTSVR21_EL1
// CHECK:   mrs	x3, PMEVCNTSVR21_EL1            // encoding: [0xa3,0xea,0x30,0xd5]
            mrs	x3, PMEVCNTSVR22_EL1
// CHECK:   mrs	x3, PMEVCNTSVR22_EL1            // encoding: [0xc3,0xea,0x30,0xd5]
            mrs	x3, PMEVCNTSVR23_EL1
// CHECK:   mrs	x3, PMEVCNTSVR23_EL1            // encoding: [0xe3,0xea,0x30,0xd5]
            mrs	x3, PMEVCNTSVR24_EL1
// CHECK:   mrs	x3, PMEVCNTSVR24_EL1            // encoding: [0x03,0xeb,0x30,0xd5]
            mrs	x3, PMEVCNTSVR25_EL1
// CHECK:   mrs	x3, PMEVCNTSVR25_EL1            // encoding: [0x23,0xeb,0x30,0xd5]
            mrs	x3, PMEVCNTSVR26_EL1
// CHECK:   mrs	x3, PMEVCNTSVR26_EL1            // encoding: [0x43,0xeb,0x30,0xd5]
            mrs	x3, PMEVCNTSVR27_EL1
// CHECK:   mrs	x3, PMEVCNTSVR27_EL1            // encoding: [0x63,0xeb,0x30,0xd5]
            mrs	x3, PMEVCNTSVR28_EL1
// CHECK:   mrs	x3, PMEVCNTSVR28_EL1            // encoding: [0x83,0xeb,0x30,0xd5]
            mrs	x3, PMEVCNTSVR29_EL1
// CHECK:   mrs	x3, PMEVCNTSVR29_EL1            // encoding: [0xa3,0xeb,0x30,0xd5]
            mrs	x3, PMEVCNTSVR30_EL1
// CHECK:   mrs	x3, PMEVCNTSVR30_EL1            // encoding: [0xc3,0xeb,0x30,0xd5]

// FEAT_PMUv3_ICNTR
            mrs x3, PMICNTR_EL0
// CHECK:   mrs x3, PMICNTR_EL0                 // encoding: [0x03,0x94,0x3b,0xd5]
            msr PMICNTR_EL0, x3
// CHECK:   msr PMICNTR_EL0, x3                 // encoding: [0x03,0x94,0x1b,0xd5]
            mrs x3, PMICFILTR_EL0
// CHECK:   mrs x3, PMICFILTR_EL0               // encoding: [0x03,0x96,0x3b,0xd5]
            msr PMICFILTR_EL0, x3
// CHECK:   msr PMICFILTR_EL0, x3               // encoding: [0x03,0x96,0x1b,0xd5]

// FEAT_PMUv3p9/FEAT_PMUV3_ICNTR
            msr PMZR_EL0, x3
// CHECK:   msr PMZR_EL0, x3                    // encoding: [0x83,0x9d,0x1b,0xd5]

// FEAT_SEBEP
            mrs	x3, PMECR_EL1
// CHECK:   mrs	x3, PMECR_EL1                   // encoding: [0xa3,0x9e,0x38,0xd5]
            msr	PMECR_EL1, x1
// CHECK:   msr	PMECR_EL1, x1                   // encoding: [0xa1,0x9e,0x18,0xd5]
            mrs	x3, PMIAR_EL1
// CHECK:   mrs	x3, PMIAR_EL1                   // encoding: [0xe3,0x9e,0x38,0xd5]
            msr	PMIAR_EL1, x1
// CHECK:   msr	PMIAR_EL1, x1                   // encoding: [0xe1,0x9e,0x18,0xd5]

// FEAT_SPMU
            mrs	x3, SPMACCESSR_EL1
// CHECK:   mrs	x3, SPMACCESSR_EL1              // encoding: [0x63,0x9d,0x30,0xd5]
            msr	SPMACCESSR_EL1, x1
// CHECK:   msr	SPMACCESSR_EL1, x1              // encoding: [0x61,0x9d,0x10,0xd5]
            mrs	x3, SPMACCESSR_EL12
// CHECK:   mrs	x3, SPMACCESSR_EL12             // encoding: [0x63,0x9d,0x35,0xd5]
            msr	SPMACCESSR_EL12, x1
// CHECK:   msr	SPMACCESSR_EL12, x1             // encoding: [0x61,0x9d,0x15,0xd5]
            mrs	x3, SPMACCESSR_EL2
// CHECK:   mrs	x3, SPMACCESSR_EL2              // encoding: [0x63,0x9d,0x34,0xd5]
            msr	SPMACCESSR_EL2, x1
// CHECK:   msr	SPMACCESSR_EL2, x1              // encoding: [0x61,0x9d,0x14,0xd5]
            mrs	x3, SPMACCESSR_EL3
// CHECK:   mrs	x3, SPMACCESSR_EL3              // encoding: [0x63,0x9d,0x36,0xd5]
            msr	SPMACCESSR_EL3, x1
// CHECK:   msr	SPMACCESSR_EL3, x1              // encoding: [0x61,0x9d,0x16,0xd5]
            mrs	x3, SPMCNTENCLR_EL0
// CHECK:   mrs	x3, SPMCNTENCLR_EL0             // encoding: [0x43,0x9c,0x33,0xd5]
            msr	SPMCNTENCLR_EL0, x1
// CHECK:   msr	SPMCNTENCLR_EL0, x1             // encoding: [0x41,0x9c,0x13,0xd5]
            mrs	x3, SPMCNTENSET_EL0
// CHECK:   mrs	x3, SPMCNTENSET_EL0             // encoding: [0x23,0x9c,0x33,0xd5]
            msr	SPMCNTENSET_EL0, x1
// CHECK:   msr	SPMCNTENSET_EL0, x1             // encoding: [0x21,0x9c,0x13,0xd5]
            mrs	x3, SPMCR_EL0
// CHECK:   mrs	x3, SPMCR_EL0                   // encoding: [0x03,0x9c,0x33,0xd5]
            msr	SPMCR_EL0, x1
// CHECK:   msr	SPMCR_EL0, x1                   // encoding: [0x01,0x9c,0x13,0xd5]
            mrs	x3, SPMDEVAFF_EL1
// CHECK:   mrs	x3, SPMDEVAFF_EL1               // encoding: [0xc3,0x9d,0x30,0xd5]
            mrs	x3, SPMDEVARCH_EL1
// CHECK:   mrs	x3, SPMDEVARCH_EL1              // encoding: [0xa3,0x9d,0x30,0xd5]

            mrs	x3, SPMEVCNTR0_EL0
// CHECK:   mrs	x3, SPMEVCNTR0_EL0              // encoding: [0x03,0xe0,0x33,0xd5]
            msr	SPMEVCNTR0_EL0, x1
// CHECK:   msr	SPMEVCNTR0_EL0, x1              // encoding: [0x01,0xe0,0x13,0xd5]
            mrs	x3, SPMEVCNTR1_EL0
// CHECK:   mrs	x3, SPMEVCNTR1_EL0              // encoding: [0x23,0xe0,0x33,0xd5]
            msr	SPMEVCNTR1_EL0, x1
// CHECK:   msr	SPMEVCNTR1_EL0, x1              // encoding: [0x21,0xe0,0x13,0xd5]
            mrs	x3, SPMEVCNTR2_EL0
// CHECK:   mrs	x3, SPMEVCNTR2_EL0              // encoding: [0x43,0xe0,0x33,0xd5]
            msr	SPMEVCNTR2_EL0, x1
// CHECK:   msr	SPMEVCNTR2_EL0, x1              // encoding: [0x41,0xe0,0x13,0xd5]
            mrs	x3, SPMEVCNTR3_EL0
// CHECK:   mrs	x3, SPMEVCNTR3_EL0              // encoding: [0x63,0xe0,0x33,0xd5]
            msr	SPMEVCNTR3_EL0, x1
// CHECK:   msr	SPMEVCNTR3_EL0, x1              // encoding: [0x61,0xe0,0x13,0xd5]
            mrs	x3, SPMEVCNTR4_EL0
// CHECK:   mrs	x3, SPMEVCNTR4_EL0              // encoding: [0x83,0xe0,0x33,0xd5]
            msr	SPMEVCNTR4_EL0, x1
// CHECK:   msr	SPMEVCNTR4_EL0, x1              // encoding: [0x81,0xe0,0x13,0xd5]
            mrs	x3, SPMEVCNTR5_EL0
// CHECK:   mrs	x3, SPMEVCNTR5_EL0              // encoding: [0xa3,0xe0,0x33,0xd5]
            msr	SPMEVCNTR5_EL0, x1
// CHECK:   msr	SPMEVCNTR5_EL0, x1              // encoding: [0xa1,0xe0,0x13,0xd5]
            mrs	x3, SPMEVCNTR6_EL0
// CHECK:   mrs	x3, SPMEVCNTR6_EL0              // encoding: [0xc3,0xe0,0x33,0xd5]
            msr	SPMEVCNTR6_EL0, x1
// CHECK:   msr	SPMEVCNTR6_EL0, x1              // encoding: [0xc1,0xe0,0x13,0xd5]
            mrs	x3, SPMEVCNTR7_EL0
// CHECK:   mrs	x3, SPMEVCNTR7_EL0              // encoding: [0xe3,0xe0,0x33,0xd5]
            msr	SPMEVCNTR7_EL0, x1
// CHECK:   msr	SPMEVCNTR7_EL0, x1              // encoding: [0xe1,0xe0,0x13,0xd5]
            mrs	x3, SPMEVCNTR8_EL0
// CHECK:   mrs	x3, SPMEVCNTR8_EL0              // encoding: [0x03,0xe1,0x33,0xd5]
            msr	SPMEVCNTR8_EL0, x1
// CHECK:   msr	SPMEVCNTR8_EL0, x1              // encoding: [0x01,0xe1,0x13,0xd5]
            mrs	x3, SPMEVCNTR9_EL0
// CHECK:   mrs	x3, SPMEVCNTR9_EL0              // encoding: [0x23,0xe1,0x33,0xd5]
            msr	SPMEVCNTR9_EL0, x1
// CHECK:   msr	SPMEVCNTR9_EL0, x1              // encoding: [0x21,0xe1,0x13,0xd5]
            mrs	x3, SPMEVCNTR10_EL0
// CHECK:   mrs	x3, SPMEVCNTR10_EL0             // encoding: [0x43,0xe1,0x33,0xd5]
            msr	SPMEVCNTR10_EL0, x1
// CHECK:   msr	SPMEVCNTR10_EL0, x1             // encoding: [0x41,0xe1,0x13,0xd5]
            mrs	x3, SPMEVCNTR11_EL0
// CHECK:   mrs	x3, SPMEVCNTR11_EL0             // encoding: [0x63,0xe1,0x33,0xd5]
            msr	SPMEVCNTR11_EL0, x1
// CHECK:   msr	SPMEVCNTR11_EL0, x1             // encoding: [0x61,0xe1,0x13,0xd5]
            mrs	x3, SPMEVCNTR12_EL0
// CHECK:   mrs	x3, SPMEVCNTR12_EL0             // encoding: [0x83,0xe1,0x33,0xd5]
            msr	SPMEVCNTR12_EL0, x1
// CHECK:   msr	SPMEVCNTR12_EL0, x1             // encoding: [0x81,0xe1,0x13,0xd5]
            mrs	x3, SPMEVCNTR13_EL0
// CHECK:   mrs	x3, SPMEVCNTR13_EL0             // encoding: [0xa3,0xe1,0x33,0xd5]
            msr	SPMEVCNTR13_EL0, x1
// CHECK:   msr	SPMEVCNTR13_EL0, x1             // encoding: [0xa1,0xe1,0x13,0xd5]
            mrs	x3, SPMEVCNTR14_EL0
// CHECK:   mrs	x3, SPMEVCNTR14_EL0             // encoding: [0xc3,0xe1,0x33,0xd5]
            msr	SPMEVCNTR14_EL0, x1
// CHECK:   msr	SPMEVCNTR14_EL0, x1             // encoding: [0xc1,0xe1,0x13,0xd5]
            mrs	x3, SPMEVCNTR15_EL0
// CHECK:   mrs	x3, SPMEVCNTR15_EL0             // encoding: [0xe3,0xe1,0x33,0xd5]
            msr	SPMEVCNTR15_EL0, x1
// CHECK:   msr	SPMEVCNTR15_EL0, x1             // encoding: [0xe1,0xe1,0x13,0xd5]

            mrs	x3, SPMEVFILT2R0_EL0
// CHECK:   mrs	x3, SPMEVFILT2R0_EL0            // encoding: [0x03,0xe6,0x33,0xd5]
            msr	SPMEVFILT2R0_EL0, x1
// CHECK:   msr	SPMEVFILT2R0_EL0, x1            // encoding: [0x01,0xe6,0x13,0xd5]
            mrs	x3, SPMEVFILT2R1_EL0
// CHECK:   mrs	x3, SPMEVFILT2R1_EL0            // encoding: [0x23,0xe6,0x33,0xd5]
            msr	SPMEVFILT2R1_EL0, x1
// CHECK:   msr	SPMEVFILT2R1_EL0, x1            // encoding: [0x21,0xe6,0x13,0xd5]
            mrs	x3, SPMEVFILT2R2_EL0
// CHECK:   mrs	x3, SPMEVFILT2R2_EL0            // encoding: [0x43,0xe6,0x33,0xd5]
            msr	SPMEVFILT2R2_EL0, x1
// CHECK:   msr	SPMEVFILT2R2_EL0, x1            // encoding: [0x41,0xe6,0x13,0xd5]
            mrs	x3, SPMEVFILT2R3_EL0
// CHECK:   mrs	x3, SPMEVFILT2R3_EL0            // encoding: [0x63,0xe6,0x33,0xd5]
            msr	SPMEVFILT2R3_EL0, x1
// CHECK:   msr	SPMEVFILT2R3_EL0, x1            // encoding: [0x61,0xe6,0x13,0xd5]
            mrs	x3, SPMEVFILT2R4_EL0
// CHECK:   mrs	x3, SPMEVFILT2R4_EL0            // encoding: [0x83,0xe6,0x33,0xd5]
            msr	SPMEVFILT2R4_EL0, x1
// CHECK:   msr	SPMEVFILT2R4_EL0, x1            // encoding: [0x81,0xe6,0x13,0xd5]
            mrs	x3, SPMEVFILT2R5_EL0
// CHECK:   mrs	x3, SPMEVFILT2R5_EL0            // encoding: [0xa3,0xe6,0x33,0xd5]
            msr	SPMEVFILT2R5_EL0, x1
// CHECK:   msr	SPMEVFILT2R5_EL0, x1            // encoding: [0xa1,0xe6,0x13,0xd5]
            mrs	x3, SPMEVFILT2R6_EL0
// CHECK:   mrs	x3, SPMEVFILT2R6_EL0            // encoding: [0xc3,0xe6,0x33,0xd5]
            msr	SPMEVFILT2R6_EL0, x1
// CHECK:   msr	SPMEVFILT2R6_EL0, x1            // encoding: [0xc1,0xe6,0x13,0xd5]
            mrs	x3, SPMEVFILT2R7_EL0
// CHECK:   mrs	x3, SPMEVFILT2R7_EL0            // encoding: [0xe3,0xe6,0x33,0xd5]
            msr	SPMEVFILT2R7_EL0, x1
// CHECK:   msr	SPMEVFILT2R7_EL0, x1            // encoding: [0xe1,0xe6,0x13,0xd5]
            mrs	x3, SPMEVFILT2R8_EL0
// CHECK:   mrs	x3, SPMEVFILT2R8_EL0            // encoding: [0x03,0xe7,0x33,0xd5]
            msr	SPMEVFILT2R8_EL0, x1
// CHECK:   msr	SPMEVFILT2R8_EL0, x1            // encoding: [0x01,0xe7,0x13,0xd5]
            mrs	x3, SPMEVFILT2R9_EL0
// CHECK:   mrs	x3, SPMEVFILT2R9_EL0            // encoding: [0x23,0xe7,0x33,0xd5]
            msr	SPMEVFILT2R9_EL0, x1
// CHECK:   msr	SPMEVFILT2R9_EL0, x1            // encoding: [0x21,0xe7,0x13,0xd5]
            mrs	x3, SPMEVFILT2R10_EL0
// CHECK:   mrs	x3, SPMEVFILT2R10_EL0           // encoding: [0x43,0xe7,0x33,0xd5]
            msr	SPMEVFILT2R10_EL0, x1
// CHECK:   msr	SPMEVFILT2R10_EL0, x1           // encoding: [0x41,0xe7,0x13,0xd5]
            mrs	x3, SPMEVFILT2R11_EL0
// CHECK:   mrs	x3, SPMEVFILT2R11_EL0           // encoding: [0x63,0xe7,0x33,0xd5]
            msr	SPMEVFILT2R11_EL0, x1
// CHECK:   msr	SPMEVFILT2R11_EL0, x1           // encoding: [0x61,0xe7,0x13,0xd5]
            mrs	x3, SPMEVFILT2R12_EL0
// CHECK:   mrs	x3, SPMEVFILT2R12_EL0           // encoding: [0x83,0xe7,0x33,0xd5]
            msr	SPMEVFILT2R12_EL0, x1
// CHECK:   msr	SPMEVFILT2R12_EL0, x1           // encoding: [0x81,0xe7,0x13,0xd5]
            mrs	x3, SPMEVFILT2R13_EL0
// CHECK:   mrs	x3, SPMEVFILT2R13_EL0           // encoding: [0xa3,0xe7,0x33,0xd5]
            msr	SPMEVFILT2R13_EL0, x1
// CHECK:   msr	SPMEVFILT2R13_EL0, x1           // encoding: [0xa1,0xe7,0x13,0xd5]
            mrs	x3, SPMEVFILT2R14_EL0
// CHECK:   mrs	x3, SPMEVFILT2R14_EL0           // encoding: [0xc3,0xe7,0x33,0xd5]
            msr	SPMEVFILT2R14_EL0, x1
// CHECK:   msr	SPMEVFILT2R14_EL0, x1           // encoding: [0xc1,0xe7,0x13,0xd5]
            mrs	x3, SPMEVFILT2R15_EL0
// CHECK:   mrs	x3, SPMEVFILT2R15_EL0           // encoding: [0xe3,0xe7,0x33,0xd5]
            msr	SPMEVFILT2R15_EL0, x1
// CHECK:   msr	SPMEVFILT2R15_EL0, x1           // encoding: [0xe1,0xe7,0x13,0xd5]

            mrs	x3, SPMEVFILTR0_EL0
// CHECK:   mrs	x3, SPMEVFILTR0_EL0             // encoding: [0x03,0xe4,0x33,0xd5]
            msr	SPMEVFILTR0_EL0, x1
// CHECK:   msr	SPMEVFILTR0_EL0, x1             // encoding: [0x01,0xe4,0x13,0xd5]
            mrs	x3, SPMEVFILTR1_EL0
// CHECK:   mrs	x3, SPMEVFILTR1_EL0             // encoding: [0x23,0xe4,0x33,0xd5]
            msr	SPMEVFILTR1_EL0, x1
// CHECK:   msr	SPMEVFILTR1_EL0, x1             // encoding: [0x21,0xe4,0x13,0xd5]
            mrs	x3, SPMEVFILTR2_EL0
// CHECK:   mrs	x3, SPMEVFILTR2_EL0             // encoding: [0x43,0xe4,0x33,0xd5]
            msr	SPMEVFILTR2_EL0, x1
// CHECK:   msr	SPMEVFILTR2_EL0, x1             // encoding: [0x41,0xe4,0x13,0xd5]
            mrs	x3, SPMEVFILTR3_EL0
// CHECK:   mrs	x3, SPMEVFILTR3_EL0             // encoding: [0x63,0xe4,0x33,0xd5]
            msr	SPMEVFILTR3_EL0, x1
// CHECK:   msr	SPMEVFILTR3_EL0, x1             // encoding: [0x61,0xe4,0x13,0xd5]
            mrs	x3, SPMEVFILTR4_EL0
// CHECK:   mrs	x3, SPMEVFILTR4_EL0             // encoding: [0x83,0xe4,0x33,0xd5]
            msr	SPMEVFILTR4_EL0, x1
// CHECK:   msr	SPMEVFILTR4_EL0, x1             // encoding: [0x81,0xe4,0x13,0xd5]
            mrs	x3, SPMEVFILTR5_EL0
// CHECK:   mrs	x3, SPMEVFILTR5_EL0             // encoding: [0xa3,0xe4,0x33,0xd5]
            msr	SPMEVFILTR5_EL0, x1
// CHECK:   msr	SPMEVFILTR5_EL0, x1             // encoding: [0xa1,0xe4,0x13,0xd5]
            mrs	x3, SPMEVFILTR6_EL0
// CHECK:   mrs	x3, SPMEVFILTR6_EL0             // encoding: [0xc3,0xe4,0x33,0xd5]
            msr	SPMEVFILTR6_EL0, x1
// CHECK:   msr	SPMEVFILTR6_EL0, x1             // encoding: [0xc1,0xe4,0x13,0xd5]
            mrs	x3, SPMEVFILTR7_EL0
// CHECK:   mrs	x3, SPMEVFILTR7_EL0             // encoding: [0xe3,0xe4,0x33,0xd5]
            msr	SPMEVFILTR7_EL0, x1
// CHECK:   msr	SPMEVFILTR7_EL0, x1             // encoding: [0xe1,0xe4,0x13,0xd5]
            mrs	x3, SPMEVFILTR8_EL0
// CHECK:   mrs	x3, SPMEVFILTR8_EL0             // encoding: [0x03,0xe5,0x33,0xd5]
            msr	SPMEVFILTR8_EL0, x1
// CHECK:   msr	SPMEVFILTR8_EL0, x1             // encoding: [0x01,0xe5,0x13,0xd5]
            mrs	x3, SPMEVFILTR9_EL0
// CHECK:   mrs	x3, SPMEVFILTR9_EL0             // encoding: [0x23,0xe5,0x33,0xd5]
            msr	SPMEVFILTR9_EL0, x1
// CHECK:   msr	SPMEVFILTR9_EL0, x1             // encoding: [0x21,0xe5,0x13,0xd5]
            mrs	x3, SPMEVFILTR10_EL0
// CHECK:   mrs	x3, SPMEVFILTR10_EL0            // encoding: [0x43,0xe5,0x33,0xd5]
            msr	SPMEVFILTR10_EL0, x1
// CHECK:   msr	SPMEVFILTR10_EL0, x1            // encoding: [0x41,0xe5,0x13,0xd5]
            mrs	x3, SPMEVFILTR11_EL0
// CHECK:   mrs	x3, SPMEVFILTR11_EL0            // encoding: [0x63,0xe5,0x33,0xd5]
            msr	SPMEVFILTR11_EL0, x1
// CHECK:   msr	SPMEVFILTR11_EL0, x1            // encoding: [0x61,0xe5,0x13,0xd5]
            mrs	x3, SPMEVFILTR12_EL0
// CHECK:   mrs	x3, SPMEVFILTR12_EL0            // encoding: [0x83,0xe5,0x33,0xd5]
            msr	SPMEVFILTR12_EL0, x1
// CHECK:   msr	SPMEVFILTR12_EL0, x1            // encoding: [0x81,0xe5,0x13,0xd5]
            mrs	x3, SPMEVFILTR13_EL0
// CHECK:   mrs	x3, SPMEVFILTR13_EL0            // encoding: [0xa3,0xe5,0x33,0xd5]
            msr	SPMEVFILTR13_EL0, x1
// CHECK:   msr	SPMEVFILTR13_EL0, x1            // encoding: [0xa1,0xe5,0x13,0xd5]
            mrs	x3, SPMEVFILTR14_EL0
// CHECK:   mrs	x3, SPMEVFILTR14_EL0            // encoding: [0xc3,0xe5,0x33,0xd5]
            msr	SPMEVFILTR14_EL0, x1
// CHECK:   msr	SPMEVFILTR14_EL0, x1            // encoding: [0xc1,0xe5,0x13,0xd5]
            mrs	x3, SPMEVFILTR15_EL0
// CHECK:   mrs	x3, SPMEVFILTR15_EL0            // encoding: [0xe3,0xe5,0x33,0xd5]
            msr	SPMEVFILTR15_EL0, x1
// CHECK:   msr	SPMEVFILTR15_EL0, x1            // encoding: [0xe1,0xe5,0x13,0xd5]

            mrs	x3, SPMEVTYPER0_EL0
// CHECK:   mrs	x3, SPMEVTYPER0_EL0             // encoding: [0x03,0xe2,0x33,0xd5]
            msr	SPMEVTYPER0_EL0, x1
// CHECK:   msr	SPMEVTYPER0_EL0, x1             // encoding: [0x01,0xe2,0x13,0xd5]
            mrs	x3, SPMEVTYPER1_EL0
// CHECK:   mrs	x3, SPMEVTYPER1_EL0             // encoding: [0x23,0xe2,0x33,0xd5]
            msr	SPMEVTYPER1_EL0, x1
// CHECK:   msr	SPMEVTYPER1_EL0, x1             // encoding: [0x21,0xe2,0x13,0xd5]
            mrs	x3, SPMEVTYPER2_EL0
// CHECK:   mrs	x3, SPMEVTYPER2_EL0             // encoding: [0x43,0xe2,0x33,0xd5]
            msr	SPMEVTYPER2_EL0, x1
// CHECK:   msr	SPMEVTYPER2_EL0, x1             // encoding: [0x41,0xe2,0x13,0xd5]
            mrs	x3, SPMEVTYPER3_EL0
// CHECK:   mrs	x3, SPMEVTYPER3_EL0             // encoding: [0x63,0xe2,0x33,0xd5]
            msr	SPMEVTYPER3_EL0, x1
// CHECK:   msr	SPMEVTYPER3_EL0, x1             // encoding: [0x61,0xe2,0x13,0xd5]
            mrs	x3, SPMEVTYPER4_EL0
// CHECK:   mrs	x3, SPMEVTYPER4_EL0             // encoding: [0x83,0xe2,0x33,0xd5]
            msr	SPMEVTYPER4_EL0, x1
// CHECK:   msr	SPMEVTYPER4_EL0, x1             // encoding: [0x81,0xe2,0x13,0xd5]
            mrs	x3, SPMEVTYPER5_EL0
// CHECK:   mrs	x3, SPMEVTYPER5_EL0             // encoding: [0xa3,0xe2,0x33,0xd5]
            msr	SPMEVTYPER5_EL0, x1
// CHECK:   msr	SPMEVTYPER5_EL0, x1             // encoding: [0xa1,0xe2,0x13,0xd5]
            mrs	x3, SPMEVTYPER6_EL0
// CHECK:   mrs	x3, SPMEVTYPER6_EL0             // encoding: [0xc3,0xe2,0x33,0xd5]
            msr	SPMEVTYPER6_EL0, x1
// CHECK:   msr	SPMEVTYPER6_EL0, x1             // encoding: [0xc1,0xe2,0x13,0xd5]
            mrs	x3, SPMEVTYPER7_EL0
// CHECK:   mrs	x3, SPMEVTYPER7_EL0             // encoding: [0xe3,0xe2,0x33,0xd5]
            msr	SPMEVTYPER7_EL0, x1
// CHECK:   msr	SPMEVTYPER7_EL0, x1             // encoding: [0xe1,0xe2,0x13,0xd5]
            mrs	x3, SPMEVTYPER8_EL0
// CHECK:   mrs	x3, SPMEVTYPER8_EL0             // encoding: [0x03,0xe3,0x33,0xd5]
            msr	SPMEVTYPER8_EL0, x1
// CHECK:   msr	SPMEVTYPER8_EL0, x1             // encoding: [0x01,0xe3,0x13,0xd5]
            mrs	x3, SPMEVTYPER9_EL0
// CHECK:   mrs	x3, SPMEVTYPER9_EL0             // encoding: [0x23,0xe3,0x33,0xd5]
            msr	SPMEVTYPER9_EL0, x1
// CHECK:   msr	SPMEVTYPER9_EL0, x1             // encoding: [0x21,0xe3,0x13,0xd5]
            mrs	x3, SPMEVTYPER10_EL0
// CHECK:   mrs	x3, SPMEVTYPER10_EL0            // encoding: [0x43,0xe3,0x33,0xd5]
            msr	SPMEVTYPER10_EL0, x1
// CHECK:   msr	SPMEVTYPER10_EL0, x1            // encoding: [0x41,0xe3,0x13,0xd5]
            mrs	x3, SPMEVTYPER11_EL0
// CHECK:   mrs	x3, SPMEVTYPER11_EL0            // encoding: [0x63,0xe3,0x33,0xd5]
            msr	SPMEVTYPER11_EL0, x1
// CHECK:   msr	SPMEVTYPER11_EL0, x1            // encoding: [0x61,0xe3,0x13,0xd5]
            mrs	x3, SPMEVTYPER12_EL0
// CHECK:   mrs	x3, SPMEVTYPER12_EL0            // encoding: [0x83,0xe3,0x33,0xd5]
            msr	SPMEVTYPER12_EL0, x1
// CHECK:   msr	SPMEVTYPER12_EL0, x1            // encoding: [0x81,0xe3,0x13,0xd5]
            mrs	x3, SPMEVTYPER13_EL0
// CHECK:   mrs	x3, SPMEVTYPER13_EL0            // encoding: [0xa3,0xe3,0x33,0xd5]
            msr	SPMEVTYPER13_EL0, x1
// CHECK:   msr	SPMEVTYPER13_EL0, x1            // encoding: [0xa1,0xe3,0x13,0xd5]
            mrs	x3, SPMEVTYPER14_EL0
// CHECK:   mrs	x3, SPMEVTYPER14_EL0            // encoding: [0xc3,0xe3,0x33,0xd5]
            msr	SPMEVTYPER14_EL0, x1
// CHECK:   msr	SPMEVTYPER14_EL0, x1            // encoding: [0xc1,0xe3,0x13,0xd5]
            mrs	x3, SPMEVTYPER15_EL0
// CHECK:   mrs	x3, SPMEVTYPER15_EL0            // encoding: [0xe3,0xe3,0x33,0xd5]
            msr	SPMEVTYPER15_EL0, x1
// CHECK:   msr	SPMEVTYPER15_EL0, x1            // encoding: [0xe1,0xe3,0x13,0xd5]

            mrs	x3, SPMIIDR_EL1
// CHECK:   mrs	x3, SPMIIDR_EL1                 // encoding: [0x83,0x9d,0x30,0xd5]
            mrs	x3, SPMINTENCLR_EL1
// CHECK:   mrs	x3, SPMINTENCLR_EL1             // encoding: [0x43,0x9e,0x30,0xd5]
            msr	SPMINTENCLR_EL1, x1
// CHECK:   msr	SPMINTENCLR_EL1, x1             // encoding: [0x41,0x9e,0x10,0xd5]
            mrs	x3, SPMINTENSET_EL1
// CHECK:   mrs	x3, SPMINTENSET_EL1             // encoding: [0x23,0x9e,0x30,0xd5]
            msr	SPMINTENSET_EL1, x1
// CHECK:   msr	SPMINTENSET_EL1, x1             // encoding: [0x21,0x9e,0x10,0xd5]
            mrs	x3, SPMOVSCLR_EL0
// CHECK:   mrs	x3, SPMOVSCLR_EL0               // encoding: [0x63,0x9c,0x33,0xd5]
            msr	SPMOVSCLR_EL0, x1
// CHECK:   msr	SPMOVSCLR_EL0, x1               // encoding: [0x61,0x9c,0x13,0xd5]
            mrs	x3, SPMOVSSET_EL0
// CHECK:   mrs	x3, SPMOVSSET_EL0               // encoding: [0x63,0x9e,0x33,0xd5]
            msr	SPMOVSSET_EL0, x1
// CHECK:   msr	SPMOVSSET_EL0, x1               // encoding: [0x61,0x9e,0x13,0xd5]
            mrs	x3, SPMSELR_EL0
// CHECK:   mrs	x3, SPMSELR_EL0                 // encoding: [0xa3,0x9c,0x33,0xd5]
            msr	SPMSELR_EL0, x1
// CHECK:   msr	SPMSELR_EL0, x1                 // encoding: [0xa1,0x9c,0x13,0xd5]
            mrs x3, SPMCGCR0_EL1
// CHECK:   mrs x3, SPMCGCR0_EL1                // encoding: [0x03,0x9d,0x30,0xd5]
            mrs x3, SPMCGCR1_EL1
// CHECK:   mrs x3, SPMCGCR1_EL1                // encoding: [0x23,0x9d,0x30,0xd5]
            mrs x3, SPMCFGR_EL1
// CHECK:   mrs x3, SPMCFGR_EL1                 // encoding: [0xe3,0x9d,0x30,0xd5]
            mrs x3, SPMROOTCR_EL3
// CHECK:   mrs x3, SPMROOTCR_EL3               // encoding: [0xe3,0x9e,0x36,0xd5]
            msr SPMROOTCR_EL3, x3
// CHECK:   msr SPMROOTCR_EL3, x3               // encoding: [0xe3,0x9e,0x16,0xd5]
            mrs x3, SPMSCR_EL1
// CHECK:   mrs x3, SPMSCR_EL1                  // encoding: [0xe3,0x9e,0x37,0xd5]
            msr SPMSCR_EL1, x3
// CHECK:   msr SPMSCR_EL1, x3                  // encoding: [0xe3,0x9e,0x17,0xd5]

// FEAT_ITE
            mrs x3, TRCITEEDCR
// CHECK:   mrs x3, TRCITEEDCR                  // encoding: [0x23,0x02,0x31,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:21: error: expected readable system register
            msr TRCITEEDCR, x3
// CHECK:   msr TRCITEEDCR, x3                  // encoding: [0x23,0x02,0x11,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:17: error: expected writable system register
            mrs	x3, TRCITECR_EL1
// CHECK:   mrs	x3, TRCITECR_EL1                // encoding: [0x63,0x12,0x38,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:21: error: expected readable system register
            msr	TRCITECR_EL1, x1
// CHECK:   msr	TRCITECR_EL1, x1                // encoding: [0x61,0x12,0x18,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:17: error: expected writable system register or pstate
            mrs	x3, TRCITECR_EL12
// CHECK:   mrs	x3, TRCITECR_EL12               // encoding: [0x63,0x12,0x3d,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:21: error: expected readable system register
            msr	TRCITECR_EL12, x1
// CHECK:   msr	TRCITECR_EL12, x1               // encoding: [0x61,0x12,0x1d,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:17: error: expected writable system register or pstate
            mrs	x3, TRCITECR_EL2
// CHECK:   mrs	x3, TRCITECR_EL2                // encoding: [0x63,0x12,0x3c,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:21: error: expected readable system register
            msr	TRCITECR_EL2, x1
// CHECK:   msr	TRCITECR_EL2, x1                // encoding: [0x61,0x12,0x1c,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:17: error: expected writable system register or pstate
            trcit x1
// CHECK:   trcit x1                            // encoding: [0xe1,0x72,0x0b,0xd5]
// ERROR-NO-ITE: [[@LINE-2]]:13: error: instruction requires: ite

// FEAT_SPE_FDS
            mrs x3, PMSDSFR_EL1
// CHECK:   mrs x3, PMSDSFR_EL1                 // encoding: [0x83,0x9a,0x38,0xd5]
            msr PMSDSFR_EL1, x3
// CHECK:   msr PMSDSFR_EL1, x3                 // encoding: [0x83,0x9a,0x18,0xd5]
