// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.1a < %s | FileCheck %s


//------------------------------------------------------------------------------
// Virtualization Host Extensions
//------------------------------------------------------------------------------
    msr TTBR1_EL2, x0
    msr CONTEXTIDR_EL2, x0
    msr CNTHV_TVAL_EL2, x0
    msr CNTHV_CVAL_EL2, x0
    msr CNTHV_CTL_EL2, x0
    msr SCTLR_EL12, x0
    msr CPACR_EL12, x0
    msr TTBR0_EL12, x0
    msr TTBR1_EL12, x0
    msr TCR_EL12, x0
    msr AFSR0_EL12, x0
    msr AFSR1_EL12, x0
    msr ESR_EL12, x0
    msr FAR_EL12, x0
    msr MAIR_EL12, x0
    msr AMAIR_EL12, x0
    msr VBAR_EL12, x0
    msr CONTEXTIDR_EL12, x0
    msr CNTKCTL_EL12, x0
    msr CNTP_TVAL_EL02, x0
    msr CNTP_CTL_EL02, x0
    msr CNTP_CVAL_EL02, x0
    msr CNTV_TVAL_EL02, x0
    msr CNTV_CTL_EL02, x0
    msr CNTV_CVAL_EL02, x0
    msr SPSR_EL12, x0
    msr ELR_EL12, x0

// CHECK:   msr TTBR1_EL2, x0           // encoding: [0x20,0x20,0x1c,0xd5]
// CHECK:   msr CONTEXTIDR_EL2, x0      // encoding: [0x20,0xd0,0x1c,0xd5]
// CHECK:   msr CNTHV_TVAL_EL2, x0      // encoding: [0x00,0xe3,0x1c,0xd5]
// CHECK:   msr CNTHV_CVAL_EL2, x0      // encoding: [0x40,0xe3,0x1c,0xd5]
// CHECK:   msr CNTHV_CTL_EL2, x0       // encoding: [0x20,0xe3,0x1c,0xd5]
// CHECK:   msr SCTLR_EL12, x0          // encoding: [0x00,0x10,0x1d,0xd5]
// CHECK:   msr CPACR_EL12, x0          // encoding: [0x40,0x10,0x1d,0xd5]
// CHECK:   msr TTBR0_EL12, x0          // encoding: [0x00,0x20,0x1d,0xd5]
// CHECK:   msr TTBR1_EL12, x0          // encoding: [0x20,0x20,0x1d,0xd5]
// CHECK:   msr TCR_EL12, x0            // encoding: [0x40,0x20,0x1d,0xd5]
// CHECK:   msr AFSR0_EL12, x0          // encoding: [0x00,0x51,0x1d,0xd5]
// CHECK:   msr AFSR1_EL12, x0          // encoding: [0x20,0x51,0x1d,0xd5]
// CHECK:   msr ESR_EL12, x0            // encoding: [0x00,0x52,0x1d,0xd5]
// CHECK:   msr FAR_EL12, x0            // encoding: [0x00,0x60,0x1d,0xd5]
// CHECK:   msr MAIR_EL12, x0           // encoding: [0x00,0xa2,0x1d,0xd5]
// CHECK:   msr AMAIR_EL12, x0          // encoding: [0x00,0xa3,0x1d,0xd5]
// CHECK:   msr VBAR_EL12, x0           // encoding: [0x00,0xc0,0x1d,0xd5]
// CHECK:   msr CONTEXTIDR_EL12, x0     // encoding: [0x20,0xd0,0x1d,0xd5]
// CHECK:   msr CNTKCTL_EL12, x0        // encoding: [0x00,0xe1,0x1d,0xd5]
// CHECK:   msr CNTP_TVAL_EL02, x0      // encoding: [0x00,0xe2,0x1d,0xd5]
// CHECK:   msr CNTP_CTL_EL02, x0       // encoding: [0x20,0xe2,0x1d,0xd5]
// CHECK:   msr CNTP_CVAL_EL02, x0      // encoding: [0x40,0xe2,0x1d,0xd5]
// CHECK:   msr CNTV_TVAL_EL02, x0      // encoding: [0x00,0xe3,0x1d,0xd5]
// CHECK:   msr CNTV_CTL_EL02, x0       // encoding: [0x20,0xe3,0x1d,0xd5]
// CHECK:   msr CNTV_CVAL_EL02, x0      // encoding: [0x40,0xe3,0x1d,0xd5]
// CHECK:   msr SPSR_EL12, x0           // encoding: [0x00,0x40,0x1d,0xd5]
// CHECK:   msr ELR_EL12, x0            // encoding: [0x20,0x40,0x1d,0xd5]
