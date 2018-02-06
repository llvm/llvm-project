 // RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding < %s | FileCheck %s

        mrs x8, icc_iar1_el1
        mrs x26, icc_iar0_el1
        mrs x2, icc_hppir1_el1
        mrs x17, icc_hppir0_el1
        mrs x29, icc_rpr_el1
        mrs x4, ich_vtr_el2
        mrs x24, ich_eisr_el2
        mrs x9, ich_elrsr_el2
        mrs x24, icc_bpr1_el1
        mrs x14, icc_bpr0_el1
        mrs x19, icc_pmr_el1
        mrs x23, icc_ctlr_el1
        mrs x20, icc_ctlr_el3
        mrs x28, icc_sre_el1
        mrs x25, icc_sre_el2
        mrs x8, icc_sre_el3
        mrs x22, icc_igrpen0_el1
        mrs x5, icc_igrpen1_el1
        mrs x7, icc_igrpen1_el3
        mrs x22, icc_seien_el1
        mrs x4, icc_ap0r0_el1
        mrs x11, icc_ap0r1_el1
        mrs x27, icc_ap0r2_el1
        mrs x21, icc_ap0r3_el1
        mrs x2, icc_ap1r0_el1
        mrs x21, icc_ap1r1_el1
        mrs x10, icc_ap1r2_el1
        mrs x27, icc_ap1r3_el1
        mrs x20, ich_ap0r0_el2
        mrs x21, ich_ap0r1_el2
        mrs x5, ich_ap0r2_el2
        mrs x4, ich_ap0r3_el2
        mrs x15, ich_ap1r0_el2
        mrs x12, ich_ap1r1_el2
        mrs x27, ich_ap1r2_el2
        mrs x20, ich_ap1r3_el2
        mrs x10, ich_hcr_el2
        mrs x27, ich_misr_el2
        mrs x6, ich_vmcr_el2
        mrs x19, ich_vseir_el2
        mrs x3, ich_lr0_el2
        mrs x1, ich_lr1_el2
        mrs x22, ich_lr2_el2
        mrs x21, ich_lr3_el2
        mrs x6, ich_lr4_el2
        mrs x10, ich_lr5_el2
        mrs x11, ich_lr6_el2
        mrs x12, ich_lr7_el2
        mrs x0, ich_lr8_el2
        mrs x21, ich_lr9_el2
        mrs x13, ich_lr10_el2
        mrs x26, ich_lr11_el2
        mrs x1, ich_lr12_el2
        mrs x8, ich_lr13_el2
        mrs x2, ich_lr14_el2
        mrs x8, ich_lr15_el2
// CHECK: mrs      x8, {{icc_iar1_el1|ICC_IAR1_EL1}}           // encoding: [0x08,0xcc,0x38,0xd5]
// CHECK: mrs      x26, {{icc_iar0_el1|ICC_IAR0_EL1}}          // encoding: [0x1a,0xc8,0x38,0xd5]
// CHECK: mrs      x2, {{icc_hppir1_el1|ICC_HPPIR1_EL1}}         // encoding: [0x42,0xcc,0x38,0xd5]
// CHECK: mrs      x17, {{icc_hppir0_el1|ICC_HPPIR0_EL1}}        // encoding: [0x51,0xc8,0x38,0xd5]
// CHECK: mrs      x29, {{icc_rpr_el1|ICC_RPR_EL1}}           // encoding: [0x7d,0xcb,0x38,0xd5]
// CHECK: mrs      x4, {{ich_vtr_el2|ICH_VTR_EL2}}            // encoding: [0x24,0xcb,0x3c,0xd5]
// CHECK: mrs      x24, {{ich_eisr_el2|ICH_EISR_EL2}}          // encoding: [0x78,0xcb,0x3c,0xd5]
// CHECK: mrs      x9, {{ich_elrsr_el2|ICH_ELRSR_EL2}}           // encoding: [0xa9,0xcb,0x3c,0xd5]
// CHECK: mrs      x24, {{icc_bpr1_el1|ICC_BPR1_EL1}}          // encoding: [0x78,0xcc,0x38,0xd5]
// CHECK: mrs      x14, {{icc_bpr0_el1|ICC_BPR0_EL1}}          // encoding: [0x6e,0xc8,0x38,0xd5]
// CHECK: mrs      x19, {{icc_pmr_el1|ICC_PMR_EL1}}           // encoding: [0x13,0x46,0x38,0xd5]
// CHECK: mrs      x23, {{icc_ctlr_el1|ICC_CTLR_EL1}}          // encoding: [0x97,0xcc,0x38,0xd5]
// CHECK: mrs      x20, {{icc_ctlr_el3|ICC_CTLR_EL3}}          // encoding: [0x94,0xcc,0x3e,0xd5]
// CHECK: mrs      x28, {{icc_sre_el1|ICC_SRE_EL1}}           // encoding: [0xbc,0xcc,0x38,0xd5]
// CHECK: mrs      x25, {{icc_sre_el2|ICC_SRE_EL2}}           // encoding: [0xb9,0xc9,0x3c,0xd5]
// CHECK: mrs      x8, {{icc_sre_el3|ICC_SRE_EL3}}            // encoding: [0xa8,0xcc,0x3e,0xd5]
// CHECK: mrs      x22, {{icc_igrpen0_el1|ICC_IGRPEN0_EL1}}       // encoding: [0xd6,0xcc,0x38,0xd5]
// CHECK: mrs      x5, {{icc_igrpen1_el1|ICC_IGRPEN1_EL1}}        // encoding: [0xe5,0xcc,0x38,0xd5]
// CHECK: mrs      x7, {{icc_igrpen1_el3|ICC_IGRPEN1_EL3}}        // encoding: [0xe7,0xcc,0x3e,0xd5]
// CHECK: mrs      x22, {{icc_seien_el1|ICC_SEIEN_EL1}}         // encoding: [0x16,0xcd,0x38,0xd5]
// CHECK: mrs      x4, {{icc_ap0r0_el1|ICC_AP0R0_EL1}}          // encoding: [0x84,0xc8,0x38,0xd5]
// CHECK: mrs      x11, {{icc_ap0r1_el1|ICC_AP0R1_EL1}}         // encoding: [0xab,0xc8,0x38,0xd5]
// CHECK: mrs      x27, {{icc_ap0r2_el1|ICC_AP0R2_EL1}}         // encoding: [0xdb,0xc8,0x38,0xd5]
// CHECK: mrs      x21, {{icc_ap0r3_el1|ICC_AP0R3_EL1}}         // encoding: [0xf5,0xc8,0x38,0xd5]
// CHECK: mrs      x2, {{icc_ap1r0_el1|ICC_AP1R0_EL1}}          // encoding: [0x02,0xc9,0x38,0xd5]
// CHECK: mrs      x21, {{icc_ap1r1_el1|ICC_AP1R1_EL1}}         // encoding: [0x35,0xc9,0x38,0xd5]
// CHECK: mrs      x10, {{icc_ap1r2_el1|ICC_AP1R2_EL1}}         // encoding: [0x4a,0xc9,0x38,0xd5]
// CHECK: mrs      x27, {{icc_ap1r3_el1|ICC_AP1R3_EL1}}         // encoding: [0x7b,0xc9,0x38,0xd5]
// CHECK: mrs      x20, {{ich_ap0r0_el2|ICH_AP0R0_EL2}}         // encoding: [0x14,0xc8,0x3c,0xd5]
// CHECK: mrs      x21, {{ich_ap0r1_el2|ICH_AP0R1_EL2}}         // encoding: [0x35,0xc8,0x3c,0xd5]
// CHECK: mrs      x5, {{ich_ap0r2_el2|ICH_AP0R2_EL2}}          // encoding: [0x45,0xc8,0x3c,0xd5]
// CHECK: mrs      x4, {{ich_ap0r3_el2|ICH_AP0R3_EL2}}          // encoding: [0x64,0xc8,0x3c,0xd5]
// CHECK: mrs      x15, {{ich_ap1r0_el2|ICH_AP1R0_EL2}}         // encoding: [0x0f,0xc9,0x3c,0xd5]
// CHECK: mrs      x12, {{ich_ap1r1_el2|ICH_AP1R1_EL2}}         // encoding: [0x2c,0xc9,0x3c,0xd5]
// CHECK: mrs      x27, {{ich_ap1r2_el2|ICH_AP1R2_EL2}}         // encoding: [0x5b,0xc9,0x3c,0xd5]
// CHECK: mrs      x20, {{ich_ap1r3_el2|ICH_AP1R3_EL2}}         // encoding: [0x74,0xc9,0x3c,0xd5]
// CHECK: mrs      x10, {{ich_hcr_el2|ICH_HCR_EL2}}           // encoding: [0x0a,0xcb,0x3c,0xd5]
// CHECK: mrs      x27, {{ich_misr_el2|ICH_MISR_EL2}}          // encoding: [0x5b,0xcb,0x3c,0xd5]
// CHECK: mrs      x6, {{ich_vmcr_el2|ICH_VMCR_EL2}}           // encoding: [0xe6,0xcb,0x3c,0xd5]
// CHECK: mrs      x19, {{ich_vseir_el2|ICH_VSEIR_EL2}}         // encoding: [0x93,0xc9,0x3c,0xd5]
// CHECK: mrs      x3, {{ich_lr0_el2|ICH_LR0_EL2}}            // encoding: [0x03,0xcc,0x3c,0xd5]
// CHECK: mrs      x1, {{ich_lr1_el2|ICH_LR1_EL2}}            // encoding: [0x21,0xcc,0x3c,0xd5]
// CHECK: mrs      x22, {{ich_lr2_el2|ICH_LR2_EL2}}           // encoding: [0x56,0xcc,0x3c,0xd5]
// CHECK: mrs      x21, {{ich_lr3_el2|ICH_LR3_EL2}}           // encoding: [0x75,0xcc,0x3c,0xd5]
// CHECK: mrs      x6, {{ich_lr4_el2|ICH_LR4_EL2}}            // encoding: [0x86,0xcc,0x3c,0xd5]
// CHECK: mrs      x10, {{ich_lr5_el2|ICH_LR5_EL2}}           // encoding: [0xaa,0xcc,0x3c,0xd5]
// CHECK: mrs      x11, {{ich_lr6_el2|ICH_LR6_EL2}}           // encoding: [0xcb,0xcc,0x3c,0xd5]
// CHECK: mrs      x12, {{ich_lr7_el2|ICH_LR7_EL2}}           // encoding: [0xec,0xcc,0x3c,0xd5]
// CHECK: mrs      x0, {{ich_lr8_el2|ICH_LR8_EL2}}            // encoding: [0x00,0xcd,0x3c,0xd5]
// CHECK: mrs      x21, {{ich_lr9_el2|ICH_LR9_EL2}}           // encoding: [0x35,0xcd,0x3c,0xd5]
// CHECK: mrs      x13, {{ich_lr10_el2|ICH_LR10_EL2}}          // encoding: [0x4d,0xcd,0x3c,0xd5]
// CHECK: mrs      x26, {{ich_lr11_el2|ICH_LR11_EL2}}          // encoding: [0x7a,0xcd,0x3c,0xd5]
// CHECK: mrs      x1, {{ich_lr12_el2|ICH_LR12_EL2}}           // encoding: [0x81,0xcd,0x3c,0xd5]
// CHECK: mrs      x8, {{ich_lr13_el2|ICH_LR13_EL2}}           // encoding: [0xa8,0xcd,0x3c,0xd5]
// CHECK: mrs      x2, {{ich_lr14_el2|ICH_LR14_EL2}}           // encoding: [0xc2,0xcd,0x3c,0xd5]
// CHECK: mrs      x8, {{ich_lr15_el2|ICH_LR15_EL2}}           // encoding: [0xe8,0xcd,0x3c,0xd5]

        msr icc_eoir1_el1, x27
        msr icc_eoir0_el1, x5
        msr icc_dir_el1, x13
        msr icc_sgi1r_el1, x21
        msr icc_asgi1r_el1, x25
        msr icc_sgi0r_el1, x28
        msr icc_bpr1_el1, x7
        msr icc_bpr0_el1, x9
        msr icc_pmr_el1, x29
        msr icc_ctlr_el1, x24
        msr icc_ctlr_el3, x0
        msr icc_sre_el1, x2
        msr icc_sre_el2, x5
        msr icc_sre_el3, x10
        msr icc_igrpen0_el1, x22
        msr icc_igrpen1_el1, x11
        msr icc_igrpen1_el3, x8
        msr icc_seien_el1, x4
        msr icc_ap0r0_el1, x27
        msr icc_ap0r1_el1, x5
        msr icc_ap0r2_el1, x20
        msr icc_ap0r3_el1, x0
        msr icc_ap1r0_el1, x2
        msr icc_ap1r1_el1, x29
        msr icc_ap1r2_el1, x23
        msr icc_ap1r3_el1, x11
        msr ich_ap0r0_el2, x2
        msr ich_ap0r1_el2, x27
        msr ich_ap0r2_el2, x7
        msr ich_ap0r3_el2, x1
        msr ich_ap1r0_el2, x7
        msr ich_ap1r1_el2, x12
        msr ich_ap1r2_el2, x14
        msr ich_ap1r3_el2, x13
        msr ich_hcr_el2, x1
        msr ich_misr_el2, x10
        msr ich_vmcr_el2, x24
        msr ich_vseir_el2, x29
        msr ich_lr0_el2, x26
        msr ich_lr1_el2, x9
        msr ich_lr2_el2, x18
        msr ich_lr3_el2, x26
        msr ich_lr4_el2, x22
        msr ich_lr5_el2, x26
        msr ich_lr6_el2, x27
        msr ich_lr7_el2, x8
        msr ich_lr8_el2, x17
        msr ich_lr9_el2, x19
        msr ich_lr10_el2, x17
        msr ich_lr11_el2, x5
        msr ich_lr12_el2, x29
        msr ich_lr13_el2, x2
        msr ich_lr14_el2, x13
        msr ich_lr15_el2, x27
// CHECK: msr      {{icc_eoir1_el1|ICC_EOIR1_EL1}}, x27         // encoding: [0x3b,0xcc,0x18,0xd5]
// CHECK: msr      {{icc_eoir0_el1|ICC_EOIR0_EL1}}, x5          // encoding: [0x25,0xc8,0x18,0xd5]
// CHECK: msr      {{icc_dir_el1|ICC_DIR_EL1}}, x13           // encoding: [0x2d,0xcb,0x18,0xd5]
// CHECK: msr      {{icc_sgi1r_el1|ICC_SGI1R_EL1}}, x21         // encoding: [0xb5,0xcb,0x18,0xd5]
// CHECK: msr      {{icc_asgi1r_el1|ICC_ASGI1R_EL1}}, x25        // encoding: [0xd9,0xcb,0x18,0xd5]
// CHECK: msr      {{icc_sgi0r_el1|ICC_SGI0R_EL1}}, x28         // encoding: [0xfc,0xcb,0x18,0xd5]
// CHECK: msr      {{icc_bpr1_el1|ICC_BPR1_EL1}}, x7           // encoding: [0x67,0xcc,0x18,0xd5]
// CHECK: msr      {{icc_bpr0_el1|ICC_BPR0_EL1}}, x9           // encoding: [0x69,0xc8,0x18,0xd5]
// CHECK: msr      {{icc_pmr_el1|ICC_PMR_EL1}}, x29           // encoding: [0x1d,0x46,0x18,0xd5]
// CHECK: msr      {{icc_ctlr_el1|ICC_CTLR_EL1}}, x24          // encoding: [0x98,0xcc,0x18,0xd5]
// CHECK: msr      {{icc_ctlr_el3|ICC_CTLR_EL3}}, x0           // encoding: [0x80,0xcc,0x1e,0xd5]
// CHECK: msr      {{icc_sre_el1|ICC_SRE_EL1}}, x2            // encoding: [0xa2,0xcc,0x18,0xd5]
// CHECK: msr      {{icc_sre_el2|ICC_SRE_EL2}}, x5            // encoding: [0xa5,0xc9,0x1c,0xd5]
// CHECK: msr      {{icc_sre_el3|ICC_SRE_EL3}}, x10           // encoding: [0xaa,0xcc,0x1e,0xd5]
// CHECK: msr      {{icc_igrpen0_el1|ICC_IGRPEN0_EL1}}, x22       // encoding: [0xd6,0xcc,0x18,0xd5]
// CHECK: msr      {{icc_igrpen1_el1|ICC_IGRPEN1_EL1}}, x11       // encoding: [0xeb,0xcc,0x18,0xd5]
// CHECK: msr      {{icc_igrpen1_el3|ICC_IGRPEN1_EL3}}, x8        // encoding: [0xe8,0xcc,0x1e,0xd5]
// CHECK: msr      {{icc_seien_el1|ICC_SEIEN_EL1}}, x4          // encoding: [0x04,0xcd,0x18,0xd5]
// CHECK: msr      {{icc_ap0r0_el1|ICC_AP0R0_EL1}}, x27         // encoding: [0x9b,0xc8,0x18,0xd5]
// CHECK: msr      {{icc_ap0r1_el1|ICC_AP0R1_EL1}}, x5          // encoding: [0xa5,0xc8,0x18,0xd5]
// CHECK: msr      {{icc_ap0r2_el1|ICC_AP0R2_EL1}}, x20         // encoding: [0xd4,0xc8,0x18,0xd5]
// CHECK: msr      {{icc_ap0r3_el1|ICC_AP0R3_EL1}}, x0          // encoding: [0xe0,0xc8,0x18,0xd5]
// CHECK: msr      {{icc_ap1r0_el1|ICC_AP1R0_EL1}}, x2          // encoding: [0x02,0xc9,0x18,0xd5]
// CHECK: msr      {{icc_ap1r1_el1|ICC_AP1R1_EL1}}, x29         // encoding: [0x3d,0xc9,0x18,0xd5]
// CHECK: msr      {{icc_ap1r2_el1|ICC_AP1R2_EL1}}, x23         // encoding: [0x57,0xc9,0x18,0xd5]
// CHECK: msr      {{icc_ap1r3_el1|ICC_AP1R3_EL1}}, x11         // encoding: [0x6b,0xc9,0x18,0xd5]
// CHECK: msr      {{ich_ap0r0_el2|ICH_AP0R0_EL2}}, x2          // encoding: [0x02,0xc8,0x1c,0xd5]
// CHECK: msr      {{ich_ap0r1_el2|ICH_AP0R1_EL2}}, x27         // encoding: [0x3b,0xc8,0x1c,0xd5]
// CHECK: msr      {{ich_ap0r2_el2|ICH_AP0R2_EL2}}, x7          // encoding: [0x47,0xc8,0x1c,0xd5]
// CHECK: msr      {{ich_ap0r3_el2|ICH_AP0R3_EL2}}, x1          // encoding: [0x61,0xc8,0x1c,0xd5]
// CHECK: msr      {{ich_ap1r0_el2|ICH_AP1R0_EL2}}, x7          // encoding: [0x07,0xc9,0x1c,0xd5]
// CHECK: msr      {{ich_ap1r1_el2|ICH_AP1R1_EL2}}, x12         // encoding: [0x2c,0xc9,0x1c,0xd5]
// CHECK: msr      {{ich_ap1r2_el2|ICH_AP1R2_EL2}}, x14         // encoding: [0x4e,0xc9,0x1c,0xd5]
// CHECK: msr      {{ich_ap1r3_el2|ICH_AP1R3_EL2}}, x13         // encoding: [0x6d,0xc9,0x1c,0xd5]
// CHECK: msr      {{ich_hcr_el2|ICH_HCR_EL2}}, x1            // encoding: [0x01,0xcb,0x1c,0xd5]
// CHECK: msr      {{ich_misr_el2|ICH_MISR_EL2}}, x10          // encoding: [0x4a,0xcb,0x1c,0xd5]
// CHECK: msr      {{ich_vmcr_el2|ICH_VMCR_EL2}}, x24          // encoding: [0xf8,0xcb,0x1c,0xd5]
// CHECK: msr      {{ich_vseir_el2|ICH_VSEIR_EL2}}, x29         // encoding: [0x9d,0xc9,0x1c,0xd5]
// CHECK: msr      {{ich_lr0_el2|ICH_LR0_EL2}}, x26           // encoding: [0x1a,0xcc,0x1c,0xd5]
// CHECK: msr      {{ich_lr1_el2|ICH_LR1_EL2}}, x9            // encoding: [0x29,0xcc,0x1c,0xd5]
// CHECK: msr      {{ich_lr2_el2|ICH_LR2_EL2}}, x18           // encoding: [0x52,0xcc,0x1c,0xd5]
// CHECK: msr      {{ich_lr3_el2|ICH_LR3_EL2}}, x26           // encoding: [0x7a,0xcc,0x1c,0xd5]
// CHECK: msr      {{ich_lr4_el2|ICH_LR4_EL2}}, x22           // encoding: [0x96,0xcc,0x1c,0xd5]
// CHECK: msr      {{ich_lr5_el2|ICH_LR5_EL2}}, x26           // encoding: [0xba,0xcc,0x1c,0xd5]
// CHECK: msr      {{ich_lr6_el2|ICH_LR6_EL2}}, x27           // encoding: [0xdb,0xcc,0x1c,0xd5]
// CHECK: msr      {{ich_lr7_el2|ICH_LR7_EL2}}, x8            // encoding: [0xe8,0xcc,0x1c,0xd5]
// CHECK: msr      {{ich_lr8_el2|ICH_LR8_EL2}}, x17           // encoding: [0x11,0xcd,0x1c,0xd5]
// CHECK: msr      {{ich_lr9_el2|ICH_LR9_EL2}}, x19           // encoding: [0x33,0xcd,0x1c,0xd5]
// CHECK: msr      {{ich_lr10_el2|ICH_LR10_EL2}}, x17          // encoding: [0x51,0xcd,0x1c,0xd5]
// CHECK: msr      {{ich_lr11_el2|ICH_LR11_EL2}}, x5           // encoding: [0x65,0xcd,0x1c,0xd5]
// CHECK: msr      {{ich_lr12_el2|ICH_LR12_EL2}}, x29          // encoding: [0x9d,0xcd,0x1c,0xd5]
// CHECK: msr      {{ich_lr13_el2|ICH_LR13_EL2}}, x2           // encoding: [0xa2,0xcd,0x1c,0xd5]
// CHECK: msr      {{ich_lr14_el2|ICH_LR14_EL2}}, x13          // encoding: [0xcd,0xcd,0x1c,0xd5]
// CHECK: msr      {{ich_lr15_el2|ICH_LR15_EL2}}, x27          // encoding: [0xfb,0xcd,0x1c,0xd5]
