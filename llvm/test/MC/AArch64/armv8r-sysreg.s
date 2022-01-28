// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+v8r -o - %s | FileCheck %s
.text
mrs x0, VSCTLR_EL2
mrs x0, MPUIR_EL1
mrs x0, MPUIR_EL2
mrs x0, PRENR_EL1
mrs x0, PRENR_EL2
mrs x0, PRSELR_EL1
mrs x0, PRSELR_EL2
mrs x0, PRBAR_EL1
mrs x0, PRBAR_EL2
mrs x0, PRLAR_EL1
mrs x0, PRLAR_EL2
mrs x0, PRBAR1_EL1
mrs x0, PRBAR2_EL1
mrs x0, PRBAR3_EL1
mrs x0, PRBAR4_EL1
mrs x0, PRBAR5_EL1
mrs x0, PRBAR6_EL1
mrs x0, PRBAR7_EL1
mrs x0, PRBAR8_EL1
mrs x0, PRBAR9_EL1
mrs x0, PRBAR10_EL1
mrs x0, PRBAR11_EL1
mrs x0, PRBAR12_EL1
mrs x0, PRBAR13_EL1
mrs x0, PRBAR14_EL1
mrs x0, PRBAR15_EL1
mrs x0, PRLAR1_EL1
mrs x0, PRLAR2_EL1
mrs x0, PRLAR3_EL1
mrs x0, PRLAR4_EL1
mrs x0, PRLAR5_EL1
mrs x0, PRLAR6_EL1
mrs x0, PRLAR7_EL1
mrs x0, PRLAR8_EL1
mrs x0, PRLAR9_EL1
mrs x0, PRLAR10_EL1
mrs x0, PRLAR11_EL1
mrs x0, PRLAR12_EL1
mrs x0, PRLAR13_EL1
mrs x0, PRLAR14_EL1
mrs x0, PRLAR15_EL1
mrs x0, PRBAR1_EL2
mrs x0, PRBAR2_EL2
mrs x0, PRBAR3_EL2
mrs x0, PRBAR4_EL2
mrs x0, PRBAR5_EL2
mrs x0, PRBAR6_EL2
mrs x0, PRBAR7_EL2
mrs x0, PRBAR8_EL2
mrs x0, PRBAR9_EL2
mrs x0, PRBAR10_EL2
mrs x0, PRBAR11_EL2
mrs x0, PRBAR12_EL2
mrs x0, PRBAR13_EL2
mrs x0, PRBAR14_EL2
mrs x0, PRBAR15_EL2
mrs x0, PRLAR1_EL2
mrs x0, PRLAR2_EL2
mrs x0, PRLAR3_EL2
mrs x0, PRLAR4_EL2
mrs x0, PRLAR5_EL2
mrs x0, PRLAR6_EL2
mrs x0, PRLAR7_EL2
mrs x0, PRLAR8_EL2
mrs x0, PRLAR9_EL2
mrs x0, PRLAR10_EL2
mrs x0, PRLAR11_EL2
mrs x0, PRLAR12_EL2
mrs x0, PRLAR13_EL2
mrs x0, PRLAR14_EL2
mrs x0, PRLAR15_EL2
mrs x30, VSCTLR_EL2
mrs x30, MPUIR_EL1
mrs x30, MPUIR_EL2
mrs x30, PRENR_EL1
mrs x30, PRENR_EL2
mrs x30, PRSELR_EL1
mrs x30, PRSELR_EL2
mrs x30, PRBAR_EL1
mrs x30, PRBAR_EL2
mrs x30, PRLAR_EL1
mrs x30, PRLAR_EL2
mrs x30, PRBAR1_EL1
mrs x30, PRBAR2_EL1
mrs x30, PRBAR3_EL1
mrs x30, PRBAR4_EL1
mrs x30, PRBAR5_EL1
mrs x30, PRBAR6_EL1
mrs x30, PRBAR7_EL1
mrs x30, PRBAR8_EL1
mrs x30, PRBAR9_EL1
mrs x30, PRBAR10_EL1
mrs x30, PRBAR11_EL1
mrs x30, PRBAR12_EL1
mrs x30, PRBAR13_EL1
mrs x30, PRBAR14_EL1
mrs x30, PRBAR15_EL1
mrs x30, PRLAR1_EL1
mrs x30, PRLAR2_EL1
mrs x30, PRLAR3_EL1
mrs x30, PRLAR4_EL1
mrs x30, PRLAR5_EL1
mrs x30, PRLAR6_EL1
mrs x30, PRLAR7_EL1
mrs x30, PRLAR8_EL1
mrs x30, PRLAR9_EL1
mrs x30, PRLAR10_EL1
mrs x30, PRLAR11_EL1
mrs x30, PRLAR12_EL1
mrs x30, PRLAR13_EL1
mrs x30, PRLAR14_EL1
mrs x30, PRLAR15_EL1
mrs x30, PRBAR1_EL2
mrs x30, PRBAR2_EL2
mrs x30, PRBAR3_EL2
mrs x30, PRBAR4_EL2
mrs x30, PRBAR5_EL2
mrs x30, PRBAR6_EL2
mrs x30, PRBAR7_EL2
mrs x30, PRBAR8_EL2
mrs x30, PRBAR9_EL2
mrs x30, PRBAR10_EL2
mrs x30, PRBAR11_EL2
mrs x30, PRBAR12_EL2
mrs x30, PRBAR13_EL2
mrs x30, PRBAR14_EL2
mrs x30, PRBAR15_EL2
mrs x30, PRLAR1_EL2
mrs x30, PRLAR2_EL2
mrs x30, PRLAR3_EL2
mrs x30, PRLAR4_EL2
mrs x30, PRLAR5_EL2
mrs x30, PRLAR6_EL2
mrs x30, PRLAR7_EL2
mrs x30, PRLAR8_EL2
mrs x30, PRLAR9_EL2
mrs x30, PRLAR10_EL2
mrs x30, PRLAR11_EL2
mrs x30, PRLAR12_EL2
mrs x30, PRLAR13_EL2
mrs x30, PRLAR14_EL2
mrs x30, PRLAR15_EL2
msr VSCTLR_EL2, x0
msr MPUIR_EL1, x0
msr MPUIR_EL2, x0
msr PRENR_EL1, x0
msr PRENR_EL2, x0
msr PRSELR_EL1, x0
msr PRSELR_EL2, x0
msr PRBAR_EL1, x0
msr PRBAR_EL2, x0
msr PRLAR_EL1, x0
msr PRLAR_EL2, x0
msr PRBAR1_EL1, x0
msr PRBAR2_EL1, x0
msr PRBAR3_EL1, x0
msr PRBAR4_EL1, x0
msr PRBAR5_EL1, x0
msr PRBAR6_EL1, x0
msr PRBAR7_EL1, x0
msr PRBAR8_EL1, x0
msr PRBAR9_EL1, x0
msr PRBAR10_EL1, x0
msr PRBAR11_EL1, x0
msr PRBAR12_EL1, x0
msr PRBAR13_EL1, x0
msr PRBAR14_EL1, x0
msr PRBAR15_EL1, x0
msr PRLAR1_EL1, x0
msr PRLAR2_EL1, x0
msr PRLAR3_EL1, x0
msr PRLAR4_EL1, x0
msr PRLAR5_EL1, x0
msr PRLAR6_EL1, x0
msr PRLAR7_EL1, x0
msr PRLAR8_EL1, x0
msr PRLAR9_EL1, x0
msr PRLAR10_EL1, x0
msr PRLAR11_EL1, x0
msr PRLAR12_EL1, x0
msr PRLAR13_EL1, x0
msr PRLAR14_EL1, x0
msr PRLAR15_EL1, x0
msr PRBAR1_EL2, x0
msr PRBAR2_EL2, x0
msr PRBAR3_EL2, x0
msr PRBAR4_EL2, x0
msr PRBAR5_EL2, x0
msr PRBAR6_EL2, x0
msr PRBAR7_EL2, x0
msr PRBAR8_EL2, x0
msr PRBAR9_EL2, x0
msr PRBAR10_EL2, x0
msr PRBAR11_EL2, x0
msr PRBAR12_EL2, x0
msr PRBAR13_EL2, x0
msr PRBAR14_EL2, x0
msr PRBAR15_EL2, x0
msr PRLAR1_EL2, x0
msr PRLAR2_EL2, x0
msr PRLAR3_EL2, x0
msr PRLAR4_EL2, x0
msr PRLAR5_EL2, x0
msr PRLAR6_EL2, x0
msr PRLAR7_EL2, x0
msr PRLAR8_EL2, x0
msr PRLAR9_EL2, x0
msr PRLAR10_EL2, x0
msr PRLAR11_EL2, x0
msr PRLAR12_EL2, x0
msr PRLAR13_EL2, x0
msr PRLAR14_EL2, x0
msr PRLAR15_EL2, x0
msr VSCTLR_EL2, x30
msr MPUIR_EL1, x30
msr MPUIR_EL2, x30
msr PRENR_EL1, x30
msr PRENR_EL2, x30
msr PRSELR_EL1, x30
msr PRSELR_EL2, x30
msr PRBAR_EL1, x30
msr PRBAR_EL2, x30
msr PRLAR_EL1, x30
msr PRLAR_EL2, x30
msr PRBAR1_EL1, x30
msr PRBAR2_EL1, x30
msr PRBAR3_EL1, x30
msr PRBAR4_EL1, x30
msr PRBAR5_EL1, x30
msr PRBAR6_EL1, x30
msr PRBAR7_EL1, x30
msr PRBAR8_EL1, x30
msr PRBAR9_EL1, x30
msr PRBAR10_EL1, x30
msr PRBAR11_EL1, x30
msr PRBAR12_EL1, x30
msr PRBAR13_EL1, x30
msr PRBAR14_EL1, x30
msr PRBAR15_EL1, x30
msr PRLAR1_EL1, x30
msr PRLAR2_EL1, x30
msr PRLAR3_EL1, x30
msr PRLAR4_EL1, x30
msr PRLAR5_EL1, x30
msr PRLAR6_EL1, x30
msr PRLAR7_EL1, x30
msr PRLAR8_EL1, x30
msr PRLAR9_EL1, x30
msr PRLAR10_EL1, x30
msr PRLAR11_EL1, x30
msr PRLAR12_EL1, x30
msr PRLAR13_EL1, x30
msr PRLAR14_EL1, x30
msr PRLAR15_EL1, x30
msr PRBAR1_EL2, x30
msr PRBAR2_EL2, x30
msr PRBAR3_EL2, x30
msr PRBAR4_EL2, x30
msr PRBAR5_EL2, x30
msr PRBAR6_EL2, x30
msr PRBAR7_EL2, x30
msr PRBAR8_EL2, x30
msr PRBAR9_EL2, x30
msr PRBAR10_EL2, x30
msr PRBAR11_EL2, x30
msr PRBAR12_EL2, x30
msr PRBAR13_EL2, x30
msr PRBAR14_EL2, x30
msr PRBAR15_EL2, x30
msr PRLAR1_EL2, x30
msr PRLAR2_EL2, x30
msr PRLAR3_EL2, x30
msr PRLAR4_EL2, x30
msr PRLAR5_EL2, x30
msr PRLAR6_EL2, x30
msr PRLAR7_EL2, x30
msr PRLAR8_EL2, x30
msr PRLAR9_EL2, x30
msr PRLAR10_EL2, x30
msr PRLAR11_EL2, x30
msr PRLAR12_EL2, x30
msr PRLAR13_EL2, x30
msr PRLAR14_EL2, x30
msr PRLAR15_EL2, x30
msr CONTEXTIDR_EL2, x0

# CHECK: 	.text
# CHECK-NEXT: 	mrs	x0, VSCTLR_EL2          // encoding: [0x00,0x20,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, MPUIR_EL1           // encoding: [0x80,0x00,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, MPUIR_EL2           // encoding: [0x80,0x00,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRENR_EL1           // encoding: [0x20,0x61,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRENR_EL2           // encoding: [0x20,0x61,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRSELR_EL1          // encoding: [0x20,0x62,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRSELR_EL2          // encoding: [0x20,0x62,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR_EL1           // encoding: [0x00,0x68,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR_EL2           // encoding: [0x00,0x68,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR_EL1           // encoding: [0x20,0x68,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR_EL2           // encoding: [0x20,0x68,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR1_EL1          // encoding: [0x80,0x68,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR2_EL1          // encoding: [0x00,0x69,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR3_EL1          // encoding: [0x80,0x69,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR4_EL1          // encoding: [0x00,0x6a,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR5_EL1          // encoding: [0x80,0x6a,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR6_EL1          // encoding: [0x00,0x6b,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR7_EL1          // encoding: [0x80,0x6b,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR8_EL1          // encoding: [0x00,0x6c,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR9_EL1          // encoding: [0x80,0x6c,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR10_EL1         // encoding: [0x00,0x6d,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR11_EL1         // encoding: [0x80,0x6d,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR12_EL1         // encoding: [0x00,0x6e,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR13_EL1         // encoding: [0x80,0x6e,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR14_EL1         // encoding: [0x00,0x6f,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR15_EL1         // encoding: [0x80,0x6f,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR1_EL1          // encoding: [0xa0,0x68,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR2_EL1          // encoding: [0x20,0x69,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR3_EL1          // encoding: [0xa0,0x69,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR4_EL1          // encoding: [0x20,0x6a,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR5_EL1          // encoding: [0xa0,0x6a,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR6_EL1          // encoding: [0x20,0x6b,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR7_EL1          // encoding: [0xa0,0x6b,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR8_EL1          // encoding: [0x20,0x6c,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR9_EL1          // encoding: [0xa0,0x6c,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR10_EL1         // encoding: [0x20,0x6d,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR11_EL1         // encoding: [0xa0,0x6d,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR12_EL1         // encoding: [0x20,0x6e,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR13_EL1         // encoding: [0xa0,0x6e,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR14_EL1         // encoding: [0x20,0x6f,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR15_EL1         // encoding: [0xa0,0x6f,0x38,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR1_EL2          // encoding: [0x80,0x68,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR2_EL2          // encoding: [0x00,0x69,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR3_EL2          // encoding: [0x80,0x69,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR4_EL2          // encoding: [0x00,0x6a,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR5_EL2          // encoding: [0x80,0x6a,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR6_EL2          // encoding: [0x00,0x6b,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR7_EL2          // encoding: [0x80,0x6b,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR8_EL2          // encoding: [0x00,0x6c,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR9_EL2          // encoding: [0x80,0x6c,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR10_EL2         // encoding: [0x00,0x6d,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR11_EL2         // encoding: [0x80,0x6d,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR12_EL2         // encoding: [0x00,0x6e,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR13_EL2         // encoding: [0x80,0x6e,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR14_EL2         // encoding: [0x00,0x6f,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRBAR15_EL2         // encoding: [0x80,0x6f,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR1_EL2          // encoding: [0xa0,0x68,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR2_EL2          // encoding: [0x20,0x69,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR3_EL2          // encoding: [0xa0,0x69,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR4_EL2          // encoding: [0x20,0x6a,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR5_EL2          // encoding: [0xa0,0x6a,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR6_EL2          // encoding: [0x20,0x6b,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR7_EL2          // encoding: [0xa0,0x6b,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR8_EL2          // encoding: [0x20,0x6c,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR9_EL2          // encoding: [0xa0,0x6c,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR10_EL2         // encoding: [0x20,0x6d,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR11_EL2         // encoding: [0xa0,0x6d,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR12_EL2         // encoding: [0x20,0x6e,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR13_EL2         // encoding: [0xa0,0x6e,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR14_EL2         // encoding: [0x20,0x6f,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x0, PRLAR15_EL2         // encoding: [0xa0,0x6f,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, VSCTLR_EL2         // encoding: [0x1e,0x20,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, MPUIR_EL1          // encoding: [0x9e,0x00,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, MPUIR_EL2          // encoding: [0x9e,0x00,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRENR_EL1          // encoding: [0x3e,0x61,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRENR_EL2          // encoding: [0x3e,0x61,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRSELR_EL1         // encoding: [0x3e,0x62,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRSELR_EL2         // encoding: [0x3e,0x62,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR_EL1          // encoding: [0x1e,0x68,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR_EL2          // encoding: [0x1e,0x68,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR_EL1          // encoding: [0x3e,0x68,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR_EL2          // encoding: [0x3e,0x68,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR1_EL1         // encoding: [0x9e,0x68,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR2_EL1         // encoding: [0x1e,0x69,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR3_EL1         // encoding: [0x9e,0x69,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR4_EL1         // encoding: [0x1e,0x6a,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR5_EL1         // encoding: [0x9e,0x6a,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR6_EL1         // encoding: [0x1e,0x6b,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR7_EL1         // encoding: [0x9e,0x6b,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR8_EL1         // encoding: [0x1e,0x6c,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR9_EL1         // encoding: [0x9e,0x6c,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR10_EL1        // encoding: [0x1e,0x6d,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR11_EL1        // encoding: [0x9e,0x6d,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR12_EL1        // encoding: [0x1e,0x6e,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR13_EL1        // encoding: [0x9e,0x6e,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR14_EL1        // encoding: [0x1e,0x6f,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR15_EL1        // encoding: [0x9e,0x6f,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR1_EL1         // encoding: [0xbe,0x68,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR2_EL1         // encoding: [0x3e,0x69,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR3_EL1         // encoding: [0xbe,0x69,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR4_EL1         // encoding: [0x3e,0x6a,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR5_EL1         // encoding: [0xbe,0x6a,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR6_EL1         // encoding: [0x3e,0x6b,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR7_EL1         // encoding: [0xbe,0x6b,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR8_EL1         // encoding: [0x3e,0x6c,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR9_EL1         // encoding: [0xbe,0x6c,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR10_EL1        // encoding: [0x3e,0x6d,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR11_EL1        // encoding: [0xbe,0x6d,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR12_EL1        // encoding: [0x3e,0x6e,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR13_EL1        // encoding: [0xbe,0x6e,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR14_EL1        // encoding: [0x3e,0x6f,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR15_EL1        // encoding: [0xbe,0x6f,0x38,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR1_EL2         // encoding: [0x9e,0x68,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR2_EL2         // encoding: [0x1e,0x69,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR3_EL2         // encoding: [0x9e,0x69,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR4_EL2         // encoding: [0x1e,0x6a,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR5_EL2         // encoding: [0x9e,0x6a,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR6_EL2         // encoding: [0x1e,0x6b,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR7_EL2         // encoding: [0x9e,0x6b,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR8_EL2         // encoding: [0x1e,0x6c,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR9_EL2         // encoding: [0x9e,0x6c,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR10_EL2        // encoding: [0x1e,0x6d,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR11_EL2        // encoding: [0x9e,0x6d,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR12_EL2        // encoding: [0x1e,0x6e,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR13_EL2        // encoding: [0x9e,0x6e,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR14_EL2        // encoding: [0x1e,0x6f,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRBAR15_EL2        // encoding: [0x9e,0x6f,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR1_EL2         // encoding: [0xbe,0x68,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR2_EL2         // encoding: [0x3e,0x69,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR3_EL2         // encoding: [0xbe,0x69,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR4_EL2         // encoding: [0x3e,0x6a,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR5_EL2         // encoding: [0xbe,0x6a,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR6_EL2         // encoding: [0x3e,0x6b,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR7_EL2         // encoding: [0xbe,0x6b,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR8_EL2         // encoding: [0x3e,0x6c,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR9_EL2         // encoding: [0xbe,0x6c,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR10_EL2        // encoding: [0x3e,0x6d,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR11_EL2        // encoding: [0xbe,0x6d,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR12_EL2        // encoding: [0x3e,0x6e,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR13_EL2        // encoding: [0xbe,0x6e,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR14_EL2        // encoding: [0x3e,0x6f,0x3c,0xd5]
# CHECK-NEXT: 	mrs	x30, PRLAR15_EL2        // encoding: [0xbe,0x6f,0x3c,0xd5]
# CHECK-NEXT: 	msr	VSCTLR_EL2, x0          // encoding: [0x00,0x20,0x1c,0xd5]
# CHECK-NEXT: 	msr	MPUIR_EL1, x0           // encoding: [0x80,0x00,0x18,0xd5]
# CHECK-NEXT: 	msr	MPUIR_EL2, x0           // encoding: [0x80,0x00,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRENR_EL1, x0           // encoding: [0x20,0x61,0x18,0xd5]
# CHECK-NEXT: 	msr	PRENR_EL2, x0           // encoding: [0x20,0x61,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRSELR_EL1, x0          // encoding: [0x20,0x62,0x18,0xd5]
# CHECK-NEXT: 	msr	PRSELR_EL2, x0          // encoding: [0x20,0x62,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR_EL1, x0           // encoding: [0x00,0x68,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR_EL2, x0           // encoding: [0x00,0x68,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR_EL1, x0           // encoding: [0x20,0x68,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR_EL2, x0           // encoding: [0x20,0x68,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR1_EL1, x0          // encoding: [0x80,0x68,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR2_EL1, x0          // encoding: [0x00,0x69,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR3_EL1, x0          // encoding: [0x80,0x69,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR4_EL1, x0          // encoding: [0x00,0x6a,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR5_EL1, x0          // encoding: [0x80,0x6a,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR6_EL1, x0          // encoding: [0x00,0x6b,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR7_EL1, x0          // encoding: [0x80,0x6b,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR8_EL1, x0          // encoding: [0x00,0x6c,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR9_EL1, x0          // encoding: [0x80,0x6c,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR10_EL1, x0         // encoding: [0x00,0x6d,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR11_EL1, x0         // encoding: [0x80,0x6d,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR12_EL1, x0         // encoding: [0x00,0x6e,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR13_EL1, x0         // encoding: [0x80,0x6e,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR14_EL1, x0         // encoding: [0x00,0x6f,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR15_EL1, x0         // encoding: [0x80,0x6f,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR1_EL1, x0          // encoding: [0xa0,0x68,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR2_EL1, x0          // encoding: [0x20,0x69,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR3_EL1, x0          // encoding: [0xa0,0x69,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR4_EL1, x0          // encoding: [0x20,0x6a,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR5_EL1, x0          // encoding: [0xa0,0x6a,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR6_EL1, x0          // encoding: [0x20,0x6b,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR7_EL1, x0          // encoding: [0xa0,0x6b,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR8_EL1, x0          // encoding: [0x20,0x6c,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR9_EL1, x0          // encoding: [0xa0,0x6c,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR10_EL1, x0         // encoding: [0x20,0x6d,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR11_EL1, x0         // encoding: [0xa0,0x6d,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR12_EL1, x0         // encoding: [0x20,0x6e,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR13_EL1, x0         // encoding: [0xa0,0x6e,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR14_EL1, x0         // encoding: [0x20,0x6f,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR15_EL1, x0         // encoding: [0xa0,0x6f,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR1_EL2, x0          // encoding: [0x80,0x68,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR2_EL2, x0          // encoding: [0x00,0x69,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR3_EL2, x0          // encoding: [0x80,0x69,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR4_EL2, x0          // encoding: [0x00,0x6a,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR5_EL2, x0          // encoding: [0x80,0x6a,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR6_EL2, x0          // encoding: [0x00,0x6b,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR7_EL2, x0          // encoding: [0x80,0x6b,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR8_EL2, x0          // encoding: [0x00,0x6c,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR9_EL2, x0          // encoding: [0x80,0x6c,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR10_EL2, x0         // encoding: [0x00,0x6d,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR11_EL2, x0         // encoding: [0x80,0x6d,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR12_EL2, x0         // encoding: [0x00,0x6e,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR13_EL2, x0         // encoding: [0x80,0x6e,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR14_EL2, x0         // encoding: [0x00,0x6f,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR15_EL2, x0         // encoding: [0x80,0x6f,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR1_EL2, x0          // encoding: [0xa0,0x68,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR2_EL2, x0          // encoding: [0x20,0x69,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR3_EL2, x0          // encoding: [0xa0,0x69,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR4_EL2, x0          // encoding: [0x20,0x6a,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR5_EL2, x0          // encoding: [0xa0,0x6a,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR6_EL2, x0          // encoding: [0x20,0x6b,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR7_EL2, x0          // encoding: [0xa0,0x6b,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR8_EL2, x0          // encoding: [0x20,0x6c,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR9_EL2, x0          // encoding: [0xa0,0x6c,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR10_EL2, x0         // encoding: [0x20,0x6d,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR11_EL2, x0         // encoding: [0xa0,0x6d,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR12_EL2, x0         // encoding: [0x20,0x6e,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR13_EL2, x0         // encoding: [0xa0,0x6e,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR14_EL2, x0         // encoding: [0x20,0x6f,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR15_EL2, x0         // encoding: [0xa0,0x6f,0x1c,0xd5]
# CHECK-NEXT: 	msr	VSCTLR_EL2, x30         // encoding: [0x1e,0x20,0x1c,0xd5]
# CHECK-NEXT: 	msr	MPUIR_EL1, x30          // encoding: [0x9e,0x00,0x18,0xd5]
# CHECK-NEXT: 	msr	MPUIR_EL2, x30          // encoding: [0x9e,0x00,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRENR_EL1, x30          // encoding: [0x3e,0x61,0x18,0xd5]
# CHECK-NEXT: 	msr	PRENR_EL2, x30          // encoding: [0x3e,0x61,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRSELR_EL1, x30         // encoding: [0x3e,0x62,0x18,0xd5]
# CHECK-NEXT: 	msr	PRSELR_EL2, x30         // encoding: [0x3e,0x62,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR_EL1, x30          // encoding: [0x1e,0x68,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR_EL2, x30          // encoding: [0x1e,0x68,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR_EL1, x30          // encoding: [0x3e,0x68,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR_EL2, x30          // encoding: [0x3e,0x68,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR1_EL1, x30         // encoding: [0x9e,0x68,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR2_EL1, x30         // encoding: [0x1e,0x69,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR3_EL1, x30         // encoding: [0x9e,0x69,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR4_EL1, x30         // encoding: [0x1e,0x6a,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR5_EL1, x30         // encoding: [0x9e,0x6a,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR6_EL1, x30         // encoding: [0x1e,0x6b,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR7_EL1, x30         // encoding: [0x9e,0x6b,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR8_EL1, x30         // encoding: [0x1e,0x6c,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR9_EL1, x30         // encoding: [0x9e,0x6c,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR10_EL1, x30        // encoding: [0x1e,0x6d,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR11_EL1, x30        // encoding: [0x9e,0x6d,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR12_EL1, x30        // encoding: [0x1e,0x6e,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR13_EL1, x30        // encoding: [0x9e,0x6e,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR14_EL1, x30        // encoding: [0x1e,0x6f,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR15_EL1, x30        // encoding: [0x9e,0x6f,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR1_EL1, x30         // encoding: [0xbe,0x68,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR2_EL1, x30         // encoding: [0x3e,0x69,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR3_EL1, x30         // encoding: [0xbe,0x69,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR4_EL1, x30         // encoding: [0x3e,0x6a,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR5_EL1, x30         // encoding: [0xbe,0x6a,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR6_EL1, x30         // encoding: [0x3e,0x6b,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR7_EL1, x30         // encoding: [0xbe,0x6b,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR8_EL1, x30         // encoding: [0x3e,0x6c,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR9_EL1, x30         // encoding: [0xbe,0x6c,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR10_EL1, x30        // encoding: [0x3e,0x6d,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR11_EL1, x30        // encoding: [0xbe,0x6d,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR12_EL1, x30        // encoding: [0x3e,0x6e,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR13_EL1, x30        // encoding: [0xbe,0x6e,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR14_EL1, x30        // encoding: [0x3e,0x6f,0x18,0xd5]
# CHECK-NEXT: 	msr	PRLAR15_EL1, x30        // encoding: [0xbe,0x6f,0x18,0xd5]
# CHECK-NEXT: 	msr	PRBAR1_EL2, x30         // encoding: [0x9e,0x68,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR2_EL2, x30         // encoding: [0x1e,0x69,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR3_EL2, x30         // encoding: [0x9e,0x69,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR4_EL2, x30         // encoding: [0x1e,0x6a,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR5_EL2, x30         // encoding: [0x9e,0x6a,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR6_EL2, x30         // encoding: [0x1e,0x6b,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR7_EL2, x30         // encoding: [0x9e,0x6b,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR8_EL2, x30         // encoding: [0x1e,0x6c,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR9_EL2, x30         // encoding: [0x9e,0x6c,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR10_EL2, x30        // encoding: [0x1e,0x6d,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR11_EL2, x30        // encoding: [0x9e,0x6d,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR12_EL2, x30        // encoding: [0x1e,0x6e,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR13_EL2, x30        // encoding: [0x9e,0x6e,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR14_EL2, x30        // encoding: [0x1e,0x6f,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRBAR15_EL2, x30        // encoding: [0x9e,0x6f,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR1_EL2, x30         // encoding: [0xbe,0x68,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR2_EL2, x30         // encoding: [0x3e,0x69,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR3_EL2, x30         // encoding: [0xbe,0x69,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR4_EL2, x30         // encoding: [0x3e,0x6a,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR5_EL2, x30         // encoding: [0xbe,0x6a,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR6_EL2, x30         // encoding: [0x3e,0x6b,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR7_EL2, x30         // encoding: [0xbe,0x6b,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR8_EL2, x30         // encoding: [0x3e,0x6c,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR9_EL2, x30         // encoding: [0xbe,0x6c,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR10_EL2, x30        // encoding: [0x3e,0x6d,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR11_EL2, x30        // encoding: [0xbe,0x6d,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR12_EL2, x30        // encoding: [0x3e,0x6e,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR13_EL2, x30        // encoding: [0xbe,0x6e,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR14_EL2, x30        // encoding: [0x3e,0x6f,0x1c,0xd5]
# CHECK-NEXT: 	msr	PRLAR15_EL2, x30        // encoding: [0xbe,0x6f,0x1c,0xd5]
# CHECK-NEXT: 	msr	CONTEXTIDR_EL2, x0      // encoding: [0x20,0xd0,0x1c,0xd5]
