// RUN: llvm-mc -triple=aarch64-none-linux-gnu -show-encoding < %s | FileCheck %s

        mrs x8, trcstatr
        mrs x9, trcidr8
        mrs x11, trcidr9
        mrs x25, trcidr10
        mrs x7, trcidr11
        mrs x7, trcidr12
        mrs x6, trcidr13
        mrs x27, trcidr0
        mrs x29, trcidr1
        mrs x4, trcidr2
        mrs x8, trcidr3
        mrs x15, trcidr4
        mrs x20, trcidr5
        mrs x6, trcidr6
        mrs x6, trcidr7
        mrs x24, trcoslsr
        mrs x18, trcpdsr
        mrs x28, trcdevaff0
        mrs x5, trcdevaff1
        mrs x5, trclsr
        mrs x11, trcauthstatus
        mrs x13, trcdevarch
        mrs x18, trcdevid
        mrs x22, trcdevtype
        mrs x14, trcpidr4
        mrs x5, trcpidr5
        mrs x5, trcpidr6
        mrs x9, trcpidr7
        mrs x15, trcpidr0
        mrs x6, trcpidr1
        mrs x11, trcpidr2
        mrs x20, trcpidr3
        mrs x17, trccidr0
        mrs x2, trccidr1
        mrs x20, trccidr2
        mrs x4, trccidr3
        mrs x11, trcprgctlr
        mrs x23, trcprocselr
        mrs x13, trcconfigr
        mrs x23, trcauxctlr
        mrs x9, trceventctl0r
        mrs x16, trceventctl1r
        mrs x4, trcstallctlr
        mrs x14, trctsctlr
        mrs x24, trcsyncpr
        mrs x28, trcccctlr
        mrs x15, trcbbctlr
        mrs x1, trctraceidr
        mrs x20, trcqctlr
        mrs x2, trcvictlr
        mrs x12, trcviiectlr
        mrs x16, trcvissctlr
        mrs x8, trcvipcssctlr
        mrs x27, trcvdctlr
        mrs x9, trcvdsacctlr
        mrs x0, trcvdarcctlr
        mrs x13, trcseqevr0
        mrs x11, trcseqevr1
        mrs x26, trcseqevr2
        mrs x14, trcseqrstevr
        mrs x4, trcseqstr
        mrs x17, trcextinselr
        mrs x21, trccntrldvr0
        mrs x10, trccntrldvr1
        mrs x20, trccntrldvr2
        mrs x5, trccntrldvr3
        mrs x17, trccntctlr0
        mrs x1, trccntctlr1
        mrs x17, trccntctlr2
        mrs x6, trccntctlr3
        mrs x28, trccntvr0
        mrs x23, trccntvr1
        mrs x9, trccntvr2
        mrs x6, trccntvr3
        mrs x24, trcimspec0
        mrs x24, trcimspec1
        mrs x15, trcimspec2
        mrs x10, trcimspec3
        mrs x29, trcimspec4
        mrs x18, trcimspec5
        mrs x29, trcimspec6
        mrs x2, trcimspec7
        mrs x8, trcrsctlr2
        mrs x0, trcrsctlr3
        mrs x12, trcrsctlr4
        mrs x26, trcrsctlr5
        mrs x29, trcrsctlr6
        mrs x17, trcrsctlr7
        mrs x0, trcrsctlr8
        mrs x1, trcrsctlr9
        mrs x17, trcrsctlr10
        mrs x21, trcrsctlr11
        mrs x1, trcrsctlr12
        mrs x8, trcrsctlr13
        mrs x24, trcrsctlr14
        mrs x0, trcrsctlr15
        mrs x2, trcrsctlr16
        mrs x29, trcrsctlr17
        mrs x22, trcrsctlr18
        mrs x6, trcrsctlr19
        mrs x26, trcrsctlr20
        mrs x26, trcrsctlr21
        mrs x4, trcrsctlr22
        mrs x12, trcrsctlr23
        mrs x1, trcrsctlr24
        mrs x0, trcrsctlr25
        mrs x17, trcrsctlr26
        mrs x8, trcrsctlr27
        mrs x10, trcrsctlr28
        mrs x25, trcrsctlr29
        mrs x12, trcrsctlr30
        mrs x11, trcrsctlr31
        mrs x18, trcssccr0
        mrs x12, trcssccr1
        mrs x3, trcssccr2
        mrs x2, trcssccr3
        mrs x21, trcssccr4
        mrs x10, trcssccr5
        mrs x22, trcssccr6
        mrs x23, trcssccr7
        mrs x23, trcsscsr0
        mrs x19, trcsscsr1
        mrs x25, trcsscsr2
        mrs x17, trcsscsr3
        mrs x19, trcsscsr4
        mrs x11, trcsscsr5
        mrs x5, trcsscsr6
        mrs x9, trcsscsr7
        mrs x1, trcsspcicr0
        mrs x12, trcsspcicr1
        mrs x21, trcsspcicr2
        mrs x11, trcsspcicr3
        mrs x3, trcsspcicr4
        mrs x9, trcsspcicr5
        mrs x5, trcsspcicr6
        mrs x2, trcsspcicr7
        mrs x26, trcpdcr
        mrs x8, trcacvr0
        mrs x15, trcacvr1
        mrs x19, trcacvr2
        mrs x8, trcacvr3
        mrs x28, trcacvr4
        mrs x3, trcacvr5
        mrs x25, trcacvr6
        mrs x24, trcacvr7
        mrs x6, trcacvr8
        mrs x3, trcacvr9
        mrs x24, trcacvr10
        mrs x3, trcacvr11
        mrs x12, trcacvr12
        mrs x9, trcacvr13
        mrs x14, trcacvr14
        mrs x3, trcacvr15
        mrs x21, trcacatr0
        mrs x26, trcacatr1
        mrs x8, trcacatr2
        mrs x22, trcacatr3
        mrs x6, trcacatr4
        mrs x29, trcacatr5
        mrs x5, trcacatr6
        mrs x18, trcacatr7
        mrs x2, trcacatr8
        mrs x19, trcacatr9
        mrs x13, trcacatr10
        mrs x25, trcacatr11
        mrs x18, trcacatr12
        mrs x29, trcacatr13
        mrs x9, trcacatr14
        mrs x18, trcacatr15
        mrs x29, trcdvcvr0
        mrs x15, trcdvcvr1
        mrs x15, trcdvcvr2
        mrs x15, trcdvcvr3
        mrs x19, trcdvcvr4
        mrs x22, trcdvcvr5
        mrs x27, trcdvcvr6
        mrs x1, trcdvcvr7
        mrs x29, trcdvcmr0
        mrs x9, trcdvcmr1
        mrs x1, trcdvcmr2
        mrs x2, trcdvcmr3
        mrs x5, trcdvcmr4
        mrs x21, trcdvcmr5
        mrs x5, trcdvcmr6
        mrs x1, trcdvcmr7
        mrs x21, trccidcvr0
        mrs x24, trccidcvr1
        mrs x24, trccidcvr2
        mrs x12, trccidcvr3
        mrs x10, trccidcvr4
        mrs x9, trccidcvr5
        mrs x6, trccidcvr6
        mrs x20, trccidcvr7
        mrs x20, trcvmidcvr0
        mrs x20, trcvmidcvr1
        mrs x26, trcvmidcvr2
        mrs x1, trcvmidcvr3
        mrs x14, trcvmidcvr4
        mrs x27, trcvmidcvr5
        mrs x29, trcvmidcvr6
        mrs x17, trcvmidcvr7
        mrs x10, trccidcctlr0
        mrs x4, trccidcctlr1
        mrs x9, trcvmidcctlr0
        mrs x11, trcvmidcctlr1
        mrs x22, trcitctrl
        mrs x23, trcclaimset
        mrs x14, trcclaimclr
// CHECK: mrs      x8, {{trcstatr|TRCSTATR}}               // encoding: [0x08,0x03,0x31,0xd5]
// CHECK: mrs      x9, {{trcidr8|TRCIDR8}}                // encoding: [0xc9,0x00,0x31,0xd5]
// CHECK: mrs      x11, {{trcidr9|TRCIDR9}}               // encoding: [0xcb,0x01,0x31,0xd5]
// CHECK: mrs      x25, {{trcidr10|TRCIDR10}}              // encoding: [0xd9,0x02,0x31,0xd5]
// CHECK: mrs      x7, {{trcidr11|TRCIDR11}}               // encoding: [0xc7,0x03,0x31,0xd5]
// CHECK: mrs      x7, {{trcidr12|TRCIDR12}}               // encoding: [0xc7,0x04,0x31,0xd5]
// CHECK: mrs      x6, {{trcidr13|TRCIDR13}}               // encoding: [0xc6,0x05,0x31,0xd5]
// CHECK: mrs      x27, {{trcidr0|TRCIDR0}}               // encoding: [0xfb,0x08,0x31,0xd5]
// CHECK: mrs      x29, {{trcidr1|TRCIDR1}}               // encoding: [0xfd,0x09,0x31,0xd5]
// CHECK: mrs      x4, {{trcidr2|TRCIDR2}}                // encoding: [0xe4,0x0a,0x31,0xd5]
// CHECK: mrs      x8, {{trcidr3|TRCIDR3}}                // encoding: [0xe8,0x0b,0x31,0xd5]
// CHECK: mrs      x15, {{trcidr4|TRCIDR4}}               // encoding: [0xef,0x0c,0x31,0xd5]
// CHECK: mrs      x20, {{trcidr5|TRCIDR5}}               // encoding: [0xf4,0x0d,0x31,0xd5]
// CHECK: mrs      x6, {{trcidr6|TRCIDR6}}                // encoding: [0xe6,0x0e,0x31,0xd5]
// CHECK: mrs      x6, {{trcidr7|TRCIDR7}}                // encoding: [0xe6,0x0f,0x31,0xd5]
// CHECK: mrs      x24, {{trcoslsr|TRCOSLSR}}              // encoding: [0x98,0x11,0x31,0xd5]
// CHECK: mrs      x18, {{trcpdsr|TRCPDSR}}               // encoding: [0x92,0x15,0x31,0xd5]
// CHECK: mrs      x28, {{trcdevaff0|TRCDEVAFF0}}            // encoding: [0xdc,0x7a,0x31,0xd5]
// CHECK: mrs      x5, {{trcdevaff1|TRCDEVAFF1}}             // encoding: [0xc5,0x7b,0x31,0xd5]
// CHECK: mrs      x5, {{trclsr|TRCLSR}}                 // encoding: [0xc5,0x7d,0x31,0xd5]
// CHECK: mrs      x11, {{trcauthstatus|TRCAUTHSTATUS}}         // encoding: [0xcb,0x7e,0x31,0xd5]
// CHECK: mrs      x13, {{trcdevarch|TRCDEVARCH}}            // encoding: [0xcd,0x7f,0x31,0xd5]
// CHECK: mrs      x18, {{trcdevid|TRCDEVID}}              // encoding: [0xf2,0x72,0x31,0xd5]
// CHECK: mrs      x22, {{trcdevtype|TRCDEVTYPE}}            // encoding: [0xf6,0x73,0x31,0xd5]
// CHECK: mrs      x14, {{trcpidr4|TRCPIDR4}}              // encoding: [0xee,0x74,0x31,0xd5]
// CHECK: mrs      x5, {{trcpidr5|TRCPIDR5}}               // encoding: [0xe5,0x75,0x31,0xd5]
// CHECK: mrs      x5, {{trcpidr6|TRCPIDR6}}               // encoding: [0xe5,0x76,0x31,0xd5]
// CHECK: mrs      x9, {{trcpidr7|TRCPIDR7}}               // encoding: [0xe9,0x77,0x31,0xd5]
// CHECK: mrs      x15, {{trcpidr0|TRCPIDR0}}              // encoding: [0xef,0x78,0x31,0xd5]
// CHECK: mrs      x6, {{trcpidr1|TRCPIDR1}}               // encoding: [0xe6,0x79,0x31,0xd5]
// CHECK: mrs      x11, {{trcpidr2|TRCPIDR2}}              // encoding: [0xeb,0x7a,0x31,0xd5]
// CHECK: mrs      x20, {{trcpidr3|TRCPIDR3}}              // encoding: [0xf4,0x7b,0x31,0xd5]
// CHECK: mrs      x17, {{trccidr0|TRCCIDR0}}              // encoding: [0xf1,0x7c,0x31,0xd5]
// CHECK: mrs      x2, {{trccidr1|TRCCIDR1}}               // encoding: [0xe2,0x7d,0x31,0xd5]
// CHECK: mrs      x20, {{trccidr2|TRCCIDR2}}              // encoding: [0xf4,0x7e,0x31,0xd5]
// CHECK: mrs      x4, {{trccidr3|TRCCIDR3}}               // encoding: [0xe4,0x7f,0x31,0xd5]
// CHECK: mrs      x11, {{trcprgctlr|TRCPRGCTLR}}            // encoding: [0x0b,0x01,0x31,0xd5]
// CHECK: mrs      x23, {{trcprocselr|TRCPROCSELR}}           // encoding: [0x17,0x02,0x31,0xd5]
// CHECK: mrs      x13, {{trcconfigr|TRCCONFIGR}}            // encoding: [0x0d,0x04,0x31,0xd5]
// CHECK: mrs      x23, {{trcauxctlr|TRCAUXCTLR}}            // encoding: [0x17,0x06,0x31,0xd5]
// CHECK: mrs      x9, {{trceventctl0r|TRCEVENTCTL0R}}          // encoding: [0x09,0x08,0x31,0xd5]
// CHECK: mrs      x16, {{trceventctl1r|TRCEVENTCTL1R}}         // encoding: [0x10,0x09,0x31,0xd5]
// CHECK: mrs      x4, {{trcstallctlr|TRCSTALLCTLR}}           // encoding: [0x04,0x0b,0x31,0xd5]
// CHECK: mrs      x14, {{trctsctlr|TRCTSCTLR}}             // encoding: [0x0e,0x0c,0x31,0xd5]
// CHECK: mrs      x24, {{trcsyncpr|TRCSYNCPR}}             // encoding: [0x18,0x0d,0x31,0xd5]
// CHECK: mrs      x28, {{trcccctlr|TRCCCCTLR}}             // encoding: [0x1c,0x0e,0x31,0xd5]
// CHECK: mrs      x15, {{trcbbctlr|TRCBBCTLR}}             // encoding: [0x0f,0x0f,0x31,0xd5]
// CHECK: mrs      x1, {{trctraceidr|TRCTRACEIDR}}            // encoding: [0x21,0x00,0x31,0xd5]
// CHECK: mrs      x20, {{trcqctlr|TRCQCTLR}}              // encoding: [0x34,0x01,0x31,0xd5]
// CHECK: mrs      x2, {{trcvictlr|TRCVICTLR}}              // encoding: [0x42,0x00,0x31,0xd5]
// CHECK: mrs      x12, {{trcviiectlr|TRCVIIECTLR}}           // encoding: [0x4c,0x01,0x31,0xd5]
// CHECK: mrs      x16, {{trcvissctlr|TRCVISSCTLR}}           // encoding: [0x50,0x02,0x31,0xd5]
// CHECK: mrs      x8, {{trcvipcssctlr|TRCVIPCSSCTLR}}          // encoding: [0x48,0x03,0x31,0xd5]
// CHECK: mrs      x27, {{trcvdctlr|TRCVDCTLR}}             // encoding: [0x5b,0x08,0x31,0xd5]
// CHECK: mrs      x9, {{trcvdsacctlr|TRCVDSACCTLR}}           // encoding: [0x49,0x09,0x31,0xd5]
// CHECK: mrs      x0, {{trcvdarcctlr|TRCVDARCCTLR}}           // encoding: [0x40,0x0a,0x31,0xd5]
// CHECK: mrs      x13, {{trcseqevr0|TRCSEQEVR0}}            // encoding: [0x8d,0x00,0x31,0xd5]
// CHECK: mrs      x11, {{trcseqevr1|TRCSEQEVR1}}            // encoding: [0x8b,0x01,0x31,0xd5]
// CHECK: mrs      x26, {{trcseqevr2|TRCSEQEVR2}}            // encoding: [0x9a,0x02,0x31,0xd5]
// CHECK: mrs      x14, {{trcseqrstevr|TRCSEQRSTEVR}}          // encoding: [0x8e,0x06,0x31,0xd5]
// CHECK: mrs      x4, {{trcseqstr|TRCSEQSTR}}              // encoding: [0x84,0x07,0x31,0xd5]
// CHECK: mrs      x17, {{trcextinselr|TRCEXTINSELR}}         // encoding: [0x91,0x08,0x31,0xd5]
// CHECK: mrs      x21, {{trccntrldvr0|TRCCNTRLDVR0}}          // encoding: [0xb5,0x00,0x31,0xd5]
// CHECK: mrs      x10, {{trccntrldvr1|TRCCNTRLDVR1}}          // encoding: [0xaa,0x01,0x31,0xd5]
// CHECK: mrs      x20, {{trccntrldvr2|TRCCNTRLDVR2}}          // encoding: [0xb4,0x02,0x31,0xd5]
// CHECK: mrs      x5, {{trccntrldvr3|TRCCNTRLDVR3}}           // encoding: [0xa5,0x03,0x31,0xd5]
// CHECK: mrs      x17, {{trccntctlr0|TRCCNTCTLR0}}           // encoding: [0xb1,0x04,0x31,0xd5]
// CHECK: mrs      x1, {{trccntctlr1|TRCCNTCTLR1}}            // encoding: [0xa1,0x05,0x31,0xd5]
// CHECK: mrs      x17, {{trccntctlr2|TRCCNTCTLR2}}           // encoding: [0xb1,0x06,0x31,0xd5]
// CHECK: mrs      x6, {{trccntctlr3|TRCCNTCTLR3}}            // encoding: [0xa6,0x07,0x31,0xd5]
// CHECK: mrs      x28, {{trccntvr0|TRCCNTVR0}}             // encoding: [0xbc,0x08,0x31,0xd5]
// CHECK: mrs      x23, {{trccntvr1|TRCCNTVR1}}             // encoding: [0xb7,0x09,0x31,0xd5]
// CHECK: mrs      x9, {{trccntvr2|TRCCNTVR2}}              // encoding: [0xa9,0x0a,0x31,0xd5]
// CHECK: mrs      x6, {{trccntvr3|TRCCNTVR3}}              // encoding: [0xa6,0x0b,0x31,0xd5]
// CHECK: mrs      x24, {{trcimspec0|TRCIMSPEC0}}            // encoding: [0xf8,0x00,0x31,0xd5]
// CHECK: mrs      x24, {{trcimspec1|TRCIMSPEC1}}            // encoding: [0xf8,0x01,0x31,0xd5]
// CHECK: mrs      x15, {{trcimspec2|TRCIMSPEC2}}            // encoding: [0xef,0x02,0x31,0xd5]
// CHECK: mrs      x10, {{trcimspec3|TRCIMSPEC3}}            // encoding: [0xea,0x03,0x31,0xd5]
// CHECK: mrs      x29, {{trcimspec4|TRCIMSPEC4}}            // encoding: [0xfd,0x04,0x31,0xd5]
// CHECK: mrs      x18, {{trcimspec5|TRCIMSPEC5}}            // encoding: [0xf2,0x05,0x31,0xd5]
// CHECK: mrs      x29, {{trcimspec6|TRCIMSPEC6}}            // encoding: [0xfd,0x06,0x31,0xd5]
// CHECK: mrs      x2, {{trcimspec7|TRCIMSPEC7}}             // encoding: [0xe2,0x07,0x31,0xd5]
// CHECK: mrs      x8, {{trcrsctlr2|TRCRSCTLR2}}             // encoding: [0x08,0x12,0x31,0xd5]
// CHECK: mrs      x0, {{trcrsctlr3|TRCRSCTLR3}}             // encoding: [0x00,0x13,0x31,0xd5]
// CHECK: mrs      x12, {{trcrsctlr4|TRCRSCTLR4}}            // encoding: [0x0c,0x14,0x31,0xd5]
// CHECK: mrs      x26, {{trcrsctlr5|TRCRSCTLR5}}            // encoding: [0x1a,0x15,0x31,0xd5]
// CHECK: mrs      x29, {{trcrsctlr6|TRCRSCTLR6}}            // encoding: [0x1d,0x16,0x31,0xd5]
// CHECK: mrs      x17, {{trcrsctlr7|TRCRSCTLR7}}            // encoding: [0x11,0x17,0x31,0xd5]
// CHECK: mrs      x0, {{trcrsctlr8|TRCRSCTLR8}}             // encoding: [0x00,0x18,0x31,0xd5]
// CHECK: mrs      x1, {{trcrsctlr9|TRCRSCTLR9}}             // encoding: [0x01,0x19,0x31,0xd5]
// CHECK: mrs      x17, {{trcrsctlr10|TRCRSCTLR10}}           // encoding: [0x11,0x1a,0x31,0xd5]
// CHECK: mrs      x21, {{trcrsctlr11|TRCRSCTLR11}}           // encoding: [0x15,0x1b,0x31,0xd5]
// CHECK: mrs      x1, {{trcrsctlr12|TRCRSCTLR12}}            // encoding: [0x01,0x1c,0x31,0xd5]
// CHECK: mrs      x8, {{trcrsctlr13|TRCRSCTLR13}}            // encoding: [0x08,0x1d,0x31,0xd5]
// CHECK: mrs      x24, {{trcrsctlr14|TRCRSCTLR14}}           // encoding: [0x18,0x1e,0x31,0xd5]
// CHECK: mrs      x0, {{trcrsctlr15|TRCRSCTLR15}}            // encoding: [0x00,0x1f,0x31,0xd5]
// CHECK: mrs      x2, {{trcrsctlr16|TRCRSCTLR16}}            // encoding: [0x22,0x10,0x31,0xd5]
// CHECK: mrs      x29, {{trcrsctlr17|TRCRSCTLR17}}           // encoding: [0x3d,0x11,0x31,0xd5]
// CHECK: mrs      x22, {{trcrsctlr18|TRCRSCTLR18}}           // encoding: [0x36,0x12,0x31,0xd5]
// CHECK: mrs      x6, {{trcrsctlr19|TRCRSCTLR19}}            // encoding: [0x26,0x13,0x31,0xd5]
// CHECK: mrs      x26, {{trcrsctlr20|TRCRSCTLR20}}           // encoding: [0x3a,0x14,0x31,0xd5]
// CHECK: mrs      x26, {{trcrsctlr21|TRCRSCTLR21}}           // encoding: [0x3a,0x15,0x31,0xd5]
// CHECK: mrs      x4, {{trcrsctlr22|TRCRSCTLR22}}            // encoding: [0x24,0x16,0x31,0xd5]
// CHECK: mrs      x12, {{trcrsctlr23|TRCRSCTLR23}}           // encoding: [0x2c,0x17,0x31,0xd5]
// CHECK: mrs      x1, {{trcrsctlr24|TRCRSCTLR24}}            // encoding: [0x21,0x18,0x31,0xd5]
// CHECK: mrs      x0, {{trcrsctlr25|TRCRSCTLR25}}            // encoding: [0x20,0x19,0x31,0xd5]
// CHECK: mrs      x17, {{trcrsctlr26|TRCRSCTLR26}}           // encoding: [0x31,0x1a,0x31,0xd5]
// CHECK: mrs      x8, {{trcrsctlr27|TRCRSCTLR27}}            // encoding: [0x28,0x1b,0x31,0xd5]
// CHECK: mrs      x10, {{trcrsctlr28|TRCRSCTLR28}}           // encoding: [0x2a,0x1c,0x31,0xd5]
// CHECK: mrs      x25, {{trcrsctlr29|TRCRSCTLR29}}           // encoding: [0x39,0x1d,0x31,0xd5]
// CHECK: mrs      x12, {{trcrsctlr30|TRCRSCTLR30}}           // encoding: [0x2c,0x1e,0x31,0xd5]
// CHECK: mrs      x11, {{trcrsctlr31|TRCRSCTLR31}}           // encoding: [0x2b,0x1f,0x31,0xd5]
// CHECK: mrs      x18, {{trcssccr0|TRCSSCCR0}}             // encoding: [0x52,0x10,0x31,0xd5]
// CHECK: mrs      x12, {{trcssccr1|TRCSSCCR1}}             // encoding: [0x4c,0x11,0x31,0xd5]
// CHECK: mrs      x3, {{trcssccr2|TRCSSCCR2}}              // encoding: [0x43,0x12,0x31,0xd5]
// CHECK: mrs      x2, {{trcssccr3|TRCSSCCR3}}              // encoding: [0x42,0x13,0x31,0xd5]
// CHECK: mrs      x21, {{trcssccr4|TRCSSCCR4}}             // encoding: [0x55,0x14,0x31,0xd5]
// CHECK: mrs      x10, {{trcssccr5|TRCSSCCR5}}             // encoding: [0x4a,0x15,0x31,0xd5]
// CHECK: mrs      x22, {{trcssccr6|TRCSSCCR6}}             // encoding: [0x56,0x16,0x31,0xd5]
// CHECK: mrs      x23, {{trcssccr7|TRCSSCCR7}}             // encoding: [0x57,0x17,0x31,0xd5]
// CHECK: mrs      x23, {{trcsscsr0|TRCSSCSR0}}             // encoding: [0x57,0x18,0x31,0xd5]
// CHECK: mrs      x19, {{trcsscsr1|TRCSSCSR1}}             // encoding: [0x53,0x19,0x31,0xd5]
// CHECK: mrs      x25, {{trcsscsr2|TRCSSCSR2}}             // encoding: [0x59,0x1a,0x31,0xd5]
// CHECK: mrs      x17, {{trcsscsr3|TRCSSCSR3}}             // encoding: [0x51,0x1b,0x31,0xd5]
// CHECK: mrs      x19, {{trcsscsr4|TRCSSCSR4}}             // encoding: [0x53,0x1c,0x31,0xd5]
// CHECK: mrs      x11, {{trcsscsr5|TRCSSCSR5}}             // encoding: [0x4b,0x1d,0x31,0xd5]
// CHECK: mrs      x5, {{trcsscsr6|TRCSSCSR6}}              // encoding: [0x45,0x1e,0x31,0xd5]
// CHECK: mrs      x9, {{trcsscsr7|TRCSSCSR7}}              // encoding: [0x49,0x1f,0x31,0xd5]
// CHECK: mrs      x1, {{trcsspcicr0|TRCSSPCICR0}}            // encoding: [0x61,0x10,0x31,0xd5]
// CHECK: mrs      x12, {{trcsspcicr1|TRCSSPCICR1}}           // encoding: [0x6c,0x11,0x31,0xd5]
// CHECK: mrs      x21, {{trcsspcicr2|TRCSSPCICR2}}           // encoding: [0x75,0x12,0x31,0xd5]
// CHECK: mrs      x11, {{trcsspcicr3|TRCSSPCICR3}}           // encoding: [0x6b,0x13,0x31,0xd5]
// CHECK: mrs      x3, {{trcsspcicr4|TRCSSPCICR4}}            // encoding: [0x63,0x14,0x31,0xd5]
// CHECK: mrs      x9, {{trcsspcicr5|TRCSSPCICR5}}            // encoding: [0x69,0x15,0x31,0xd5]
// CHECK: mrs      x5, {{trcsspcicr6|TRCSSPCICR6}}            // encoding: [0x65,0x16,0x31,0xd5]
// CHECK: mrs      x2, {{trcsspcicr7|TRCSSPCICR7}}            // encoding: [0x62,0x17,0x31,0xd5]
// CHECK: mrs      x26, {{trcpdcr|TRCPDCR}}               // encoding: [0x9a,0x14,0x31,0xd5]
// CHECK: mrs      x8, {{trcacvr0|TRCACVR0}}               // encoding: [0x08,0x20,0x31,0xd5]
// CHECK: mrs      x15, {{trcacvr1|TRCACVR1}}              // encoding: [0x0f,0x22,0x31,0xd5]
// CHECK: mrs      x19, {{trcacvr2|TRCACVR2}}              // encoding: [0x13,0x24,0x31,0xd5]
// CHECK: mrs      x8, {{trcacvr3|TRCACVR3}}               // encoding: [0x08,0x26,0x31,0xd5]
// CHECK: mrs      x28, {{trcacvr4|TRCACVR4}}              // encoding: [0x1c,0x28,0x31,0xd5]
// CHECK: mrs      x3, {{trcacvr5|TRCACVR5}}               // encoding: [0x03,0x2a,0x31,0xd5]
// CHECK: mrs      x25, {{trcacvr6|TRCACVR6}}              // encoding: [0x19,0x2c,0x31,0xd5]
// CHECK: mrs      x24, {{trcacvr7|TRCACVR7}}              // encoding: [0x18,0x2e,0x31,0xd5]
// CHECK: mrs      x6, {{trcacvr8|TRCACVR8}}               // encoding: [0x26,0x20,0x31,0xd5]
// CHECK: mrs      x3, {{trcacvr9|TRCACVR9}}               // encoding: [0x23,0x22,0x31,0xd5]
// CHECK: mrs      x24, {{trcacvr10|TRCACVR10}}             // encoding: [0x38,0x24,0x31,0xd5]
// CHECK: mrs      x3, {{trcacvr11|TRCACVR11}}              // encoding: [0x23,0x26,0x31,0xd5]
// CHECK: mrs      x12, {{trcacvr12|TRCACVR12}}             // encoding: [0x2c,0x28,0x31,0xd5]
// CHECK: mrs      x9, {{trcacvr13|TRCACVR13}}              // encoding: [0x29,0x2a,0x31,0xd5]
// CHECK: mrs      x14, {{trcacvr14|TRCACVR14}}             // encoding: [0x2e,0x2c,0x31,0xd5]
// CHECK: mrs      x3, {{trcacvr15|TRCACVR15}}              // encoding: [0x23,0x2e,0x31,0xd5]
// CHECK: mrs      x21, {{trcacatr0|TRCACATR0}}             // encoding: [0x55,0x20,0x31,0xd5]
// CHECK: mrs      x26, {{trcacatr1|TRCACATR1}}             // encoding: [0x5a,0x22,0x31,0xd5]
// CHECK: mrs      x8, {{trcacatr2|TRCACATR2}}              // encoding: [0x48,0x24,0x31,0xd5]
// CHECK: mrs      x22, {{trcacatr3|TRCACATR3}}             // encoding: [0x56,0x26,0x31,0xd5]
// CHECK: mrs      x6, {{trcacatr4|TRCACATR4}}              // encoding: [0x46,0x28,0x31,0xd5]
// CHECK: mrs      x29, {{trcacatr5|TRCACATR5}}             // encoding: [0x5d,0x2a,0x31,0xd5]
// CHECK: mrs      x5, {{trcacatr6|TRCACATR6}}              // encoding: [0x45,0x2c,0x31,0xd5]
// CHECK: mrs      x18, {{trcacatr7|TRCACATR7}}             // encoding: [0x52,0x2e,0x31,0xd5]
// CHECK: mrs      x2, {{trcacatr8|TRCACATR8}}              // encoding: [0x62,0x20,0x31,0xd5]
// CHECK: mrs      x19, {{trcacatr9|TRCACATR9}}             // encoding: [0x73,0x22,0x31,0xd5]
// CHECK: mrs      x13, {{trcacatr10|TRCACATR10}}            // encoding: [0x6d,0x24,0x31,0xd5]
// CHECK: mrs      x25, {{trcacatr11|TRCACATR11}}            // encoding: [0x79,0x26,0x31,0xd5]
// CHECK: mrs      x18, {{trcacatr12|TRCACATR12}}            // encoding: [0x72,0x28,0x31,0xd5]
// CHECK: mrs      x29, {{trcacatr13|TRCACATR13}}            // encoding: [0x7d,0x2a,0x31,0xd5]
// CHECK: mrs      x9, {{trcacatr14|TRCACATR14}}             // encoding: [0x69,0x2c,0x31,0xd5]
// CHECK: mrs      x18, {{trcacatr15|TRCACATR15}}            // encoding: [0x72,0x2e,0x31,0xd5]
// CHECK: mrs      x29, {{trcdvcvr0|TRCDVCVR0}}             // encoding: [0x9d,0x20,0x31,0xd5]
// CHECK: mrs      x15, {{trcdvcvr1|TRCDVCVR1}}             // encoding: [0x8f,0x24,0x31,0xd5]
// CHECK: mrs      x15, {{trcdvcvr2|TRCDVCVR2}}             // encoding: [0x8f,0x28,0x31,0xd5]
// CHECK: mrs      x15, {{trcdvcvr3|TRCDVCVR3}}             // encoding: [0x8f,0x2c,0x31,0xd5]
// CHECK: mrs      x19, {{trcdvcvr4|TRCDVCVR4}}             // encoding: [0xb3,0x20,0x31,0xd5]
// CHECK: mrs      x22, {{trcdvcvr5|TRCDVCVR5}}             // encoding: [0xb6,0x24,0x31,0xd5]
// CHECK: mrs      x27, {{trcdvcvr6|TRCDVCVR6}}             // encoding: [0xbb,0x28,0x31,0xd5]
// CHECK: mrs      x1, {{trcdvcvr7|TRCDVCVR7}}              // encoding: [0xa1,0x2c,0x31,0xd5]
// CHECK: mrs      x29, {{trcdvcmr0|TRCDVCMR0}}             // encoding: [0xdd,0x20,0x31,0xd5]
// CHECK: mrs      x9, {{trcdvcmr1|TRCDVCMR1}}              // encoding: [0xc9,0x24,0x31,0xd5]
// CHECK: mrs      x1, {{trcdvcmr2|TRCDVCMR2}}              // encoding: [0xc1,0x28,0x31,0xd5]
// CHECK: mrs      x2, {{trcdvcmr3|TRCDVCMR3}}              // encoding: [0xc2,0x2c,0x31,0xd5]
// CHECK: mrs      x5, {{trcdvcmr4|TRCDVCMR4}}              // encoding: [0xe5,0x20,0x31,0xd5]
// CHECK: mrs      x21, {{trcdvcmr5|TRCDVCMR5}}             // encoding: [0xf5,0x24,0x31,0xd5]
// CHECK: mrs      x5, {{trcdvcmr6|TRCDVCMR6}}              // encoding: [0xe5,0x28,0x31,0xd5]
// CHECK: mrs      x1, {{trcdvcmr7|TRCDVCMR7}}              // encoding: [0xe1,0x2c,0x31,0xd5]
// CHECK: mrs      x21, {{trccidcvr0|TRCCIDCVR0}}            // encoding: [0x15,0x30,0x31,0xd5]
// CHECK: mrs      x24, {{trccidcvr1|TRCCIDCVR1}}            // encoding: [0x18,0x32,0x31,0xd5]
// CHECK: mrs      x24, {{trccidcvr2|TRCCIDCVR2}}            // encoding: [0x18,0x34,0x31,0xd5]
// CHECK: mrs      x12, {{trccidcvr3|TRCCIDCVR3}}            // encoding: [0x0c,0x36,0x31,0xd5]
// CHECK: mrs      x10, {{trccidcvr4|TRCCIDCVR4}}            // encoding: [0x0a,0x38,0x31,0xd5]
// CHECK: mrs      x9, {{trccidcvr5|TRCCIDCVR5}}             // encoding: [0x09,0x3a,0x31,0xd5]
// CHECK: mrs      x6, {{trccidcvr6|TRCCIDCVR6}}             // encoding: [0x06,0x3c,0x31,0xd5]
// CHECK: mrs      x20, {{trccidcvr7|TRCCIDCVR7}}            // encoding: [0x14,0x3e,0x31,0xd5]
// CHECK: mrs      x20, {{trcvmidcvr0|TRCVMIDCVR0}}           // encoding: [0x34,0x30,0x31,0xd5]
// CHECK: mrs      x20, {{trcvmidcvr1|TRCVMIDCVR1}}           // encoding: [0x34,0x32,0x31,0xd5]
// CHECK: mrs      x26, {{trcvmidcvr2|TRCVMIDCVR2}}           // encoding: [0x3a,0x34,0x31,0xd5]
// CHECK: mrs      x1, {{trcvmidcvr3|TRCVMIDCVR3}}            // encoding: [0x21,0x36,0x31,0xd5]
// CHECK: mrs      x14, {{trcvmidcvr4|TRCVMIDCVR4}}           // encoding: [0x2e,0x38,0x31,0xd5]
// CHECK: mrs      x27, {{trcvmidcvr5|TRCVMIDCVR5}}           // encoding: [0x3b,0x3a,0x31,0xd5]
// CHECK: mrs      x29, {{trcvmidcvr6|TRCVMIDCVR6}}           // encoding: [0x3d,0x3c,0x31,0xd5]
// CHECK: mrs      x17, {{trcvmidcvr7|TRCVMIDCVR7}}           // encoding: [0x31,0x3e,0x31,0xd5]
// CHECK: mrs      x10, {{trccidcctlr0|TRCCIDCCTLR0}}          // encoding: [0x4a,0x30,0x31,0xd5]
// CHECK: mrs      x4, {{trccidcctlr1|TRCCIDCCTLR1}}           // encoding: [0x44,0x31,0x31,0xd5]
// CHECK: mrs      x9, {{trcvmidcctlr0|TRCVMIDCCTLR0}}          // encoding: [0x49,0x32,0x31,0xd5]
// CHECK: mrs      x11, {{trcvmidcctlr1|TRCVMIDCCTLR1}}         // encoding: [0x4b,0x33,0x31,0xd5]
// CHECK: mrs      x22, {{trcitctrl|TRCITCTRL}}             // encoding: [0x96,0x70,0x31,0xd5]
// CHECK: mrs      x23, {{trcclaimset|TRCCLAIMSET}}           // encoding: [0xd7,0x78,0x31,0xd5]
// CHECK: mrs      x14, {{trcclaimclr|TRCCLAIMCLR}}           // encoding: [0xce,0x79,0x31,0xd5]

        msr trcoslar, x28
        msr trclar, x14
        msr trcprgctlr, x10
        msr trcprocselr, x27
        msr trcconfigr, x24
        msr trcauxctlr, x8
        msr trceventctl0r, x16
        msr trceventctl1r, x27
        msr trcstallctlr, x26
        msr trctsctlr, x0
        msr trcsyncpr, x14
        msr trcccctlr, x8
        msr trcbbctlr, x6
        msr trctraceidr, x23
        msr trcqctlr, x5
        msr trcvictlr, x0
        msr trcviiectlr, x0
        msr trcvissctlr, x1
        msr trcvipcssctlr, x0
        msr trcvdctlr, x7
        msr trcvdsacctlr, x18
        msr trcvdarcctlr, x24
        msr trcseqevr0, x28
        msr trcseqevr1, x21
        msr trcseqevr2, x16
        msr trcseqrstevr, x16
        msr trcseqstr, x25
        msr trcextinselr, x29
        msr trccntrldvr0, x20
        msr trccntrldvr1, x20
        msr trccntrldvr2, x22
        msr trccntrldvr3, x12
        msr trccntctlr0, x20
        msr trccntctlr1, x4
        msr trccntctlr2, x8
        msr trccntctlr3, x16
        msr trccntvr0, x5
        msr trccntvr1, x27
        msr trccntvr2, x21
        msr trccntvr3, x8
        msr trcimspec0, x6
        msr trcimspec1, x27
        msr trcimspec2, x23
        msr trcimspec3, x15
        msr trcimspec4, x13
        msr trcimspec5, x25
        msr trcimspec6, x19
        msr trcimspec7, x27
        msr trcrsctlr2, x4
        msr trcrsctlr3, x0
        msr trcrsctlr4, x21
        msr trcrsctlr5, x8
        msr trcrsctlr6, x20
        msr trcrsctlr7, x11
        msr trcrsctlr8, x18
        msr trcrsctlr9, x24
        msr trcrsctlr10, x15
        msr trcrsctlr11, x21
        msr trcrsctlr12, x4
        msr trcrsctlr13, x28
        msr trcrsctlr14, x3
        msr trcrsctlr15, x20
        msr trcrsctlr16, x12
        msr trcrsctlr17, x17
        msr trcrsctlr18, x10
        msr trcrsctlr19, x11
        msr trcrsctlr20, x3
        msr trcrsctlr21, x18
        msr trcrsctlr22, x26
        msr trcrsctlr23, x5
        msr trcrsctlr24, x25
        msr trcrsctlr25, x5
        msr trcrsctlr26, x4
        msr trcrsctlr27, x20
        msr trcrsctlr28, x5
        msr trcrsctlr29, x10
        msr trcrsctlr30, x24
        msr trcrsctlr31, x20
        msr trcssccr0, x23
        msr trcssccr1, x27
        msr trcssccr2, x27
        msr trcssccr3, x6
        msr trcssccr4, x3
        msr trcssccr5, x12
        msr trcssccr6, x7
        msr trcssccr7, x6
        msr trcsscsr0, x20
        msr trcsscsr1, x17
        msr trcsscsr2, x11
        msr trcsscsr3, x4
        msr trcsscsr4, x14
        msr trcsscsr5, x22
        msr trcsscsr6, x3
        msr trcsscsr7, x11
        msr trcsspcicr0, x2
        msr trcsspcicr1, x3
        msr trcsspcicr2, x5
        msr trcsspcicr3, x7
        msr trcsspcicr4, x11
        msr trcsspcicr5, x13
        msr trcsspcicr6, x17
        msr trcsspcicr7, x23
        msr trcpdcr, x3
        msr trcacvr0, x6
        msr trcacvr1, x20
        msr trcacvr2, x25
        msr trcacvr3, x1
        msr trcacvr4, x28
        msr trcacvr5, x15
        msr trcacvr6, x25
        msr trcacvr7, x12
        msr trcacvr8, x5
        msr trcacvr9, x25
        msr trcacvr10, x13
        msr trcacvr11, x10
        msr trcacvr12, x19
        msr trcacvr13, x10
        msr trcacvr14, x19
        msr trcacvr15, x2
        msr trcacatr0, x15
        msr trcacatr1, x13
        msr trcacatr2, x8
        msr trcacatr3, x1
        msr trcacatr4, x11
        msr trcacatr5, x8
        msr trcacatr6, x24
        msr trcacatr7, x6
        msr trcacatr8, x23
        msr trcacatr9, x5
        msr trcacatr10, x11
        msr trcacatr11, x11
        msr trcacatr12, x3
        msr trcacatr13, x28
        msr trcacatr14, x25
        msr trcacatr15, x4
        msr trcdvcvr0, x6
        msr trcdvcvr1, x3
        msr trcdvcvr2, x5
        msr trcdvcvr3, x11
        msr trcdvcvr4, x9
        msr trcdvcvr5, x14
        msr trcdvcvr6, x10
        msr trcdvcvr7, x12
        msr trcdvcmr0, x8
        msr trcdvcmr1, x8
        msr trcdvcmr2, x22
        msr trcdvcmr3, x22
        msr trcdvcmr4, x5
        msr trcdvcmr5, x16
        msr trcdvcmr6, x27
        msr trcdvcmr7, x21
        msr trccidcvr0, x8
        msr trccidcvr1, x6
        msr trccidcvr2, x9
        msr trccidcvr3, x8
        msr trccidcvr4, x3
        msr trccidcvr5, x21
        msr trccidcvr6, x12
        msr trccidcvr7, x7
        msr trcvmidcvr0, x4
        msr trcvmidcvr1, x3
        msr trcvmidcvr2, x9
        msr trcvmidcvr3, x17
        msr trcvmidcvr4, x14
        msr trcvmidcvr5, x12
        msr trcvmidcvr6, x10
        msr trcvmidcvr7, x3
        msr trccidcctlr0, x14
        msr trccidcctlr1, x22
        msr trcvmidcctlr0, x8
        msr trcvmidcctlr1, x15
        msr trcitctrl, x1
        msr trcclaimset, x7
        msr trcclaimclr, x29
// CHECK: msr      {{trcoslar|TRCOSLAR}}, x28              // encoding: [0x9c,0x10,0x11,0xd5]
// CHECK: msr      {{trclar|TRCLAR}}, x14                // encoding: [0xce,0x7c,0x11,0xd5]
// CHECK: msr      {{trcprgctlr|TRCPRGCTLR}}, x10            // encoding: [0x0a,0x01,0x11,0xd5]
// CHECK: msr      {{trcprocselr|TRCPROCSELR}}, x27           // encoding: [0x1b,0x02,0x11,0xd5]
// CHECK: msr      {{trcconfigr|TRCCONFIGR}}, x24            // encoding: [0x18,0x04,0x11,0xd5]
// CHECK: msr      {{trcauxctlr|TRCAUXCTLR}}, x8             // encoding: [0x08,0x06,0x11,0xd5]
// CHECK: msr      {{trceventctl0r|TRCEVENTCTL0R}}, x16         // encoding: [0x10,0x08,0x11,0xd5]
// CHECK: msr      {{trceventctl1r|TRCEVENTCTL1R}}, x27         // encoding: [0x1b,0x09,0x11,0xd5]
// CHECK: msr      {{trcstallctlr|TRCSTALLCTLR}}, x26          // encoding: [0x1a,0x0b,0x11,0xd5]
// CHECK: msr      {{trctsctlr|TRCTSCTLR}}, x0              // encoding: [0x00,0x0c,0x11,0xd5]
// CHECK: msr      {{trcsyncpr|TRCSYNCPR}}, x14             // encoding: [0x0e,0x0d,0x11,0xd5]
// CHECK: msr      {{trcccctlr|TRCCCCTLR}}, x8              // encoding: [0x08,0x0e,0x11,0xd5]
// CHECK: msr      {{trcbbctlr|TRCBBCTLR}}, x6              // encoding: [0x06,0x0f,0x11,0xd5]
// CHECK: msr      {{trctraceidr|TRCTRACEIDR}}, x23           // encoding: [0x37,0x00,0x11,0xd5]
// CHECK: msr      {{trcqctlr|TRCQCTLR}}, x5               // encoding: [0x25,0x01,0x11,0xd5]
// CHECK: msr      {{trcvictlr|TRCVICTLR}}, x0              // encoding: [0x40,0x00,0x11,0xd5]
// CHECK: msr      {{trcviiectlr|TRCVIIECTLR}}, x0            // encoding: [0x40,0x01,0x11,0xd5]
// CHECK: msr      {{trcvissctlr|TRCVISSCTLR}}, x1            // encoding: [0x41,0x02,0x11,0xd5]
// CHECK: msr      {{trcvipcssctlr|TRCVIPCSSCTLR}}, x0          // encoding: [0x40,0x03,0x11,0xd5]
// CHECK: msr      {{trcvdctlr|TRCVDCTLR}}, x7              // encoding: [0x47,0x08,0x11,0xd5]
// CHECK: msr      {{trcvdsacctlr|TRCVDSACCTLR}}, x18          // encoding: [0x52,0x09,0x11,0xd5]
// CHECK: msr      {{trcvdarcctlr|TRCVDARCCTLR}}, x24          // encoding: [0x58,0x0a,0x11,0xd5]
// CHECK: msr      {{trcseqevr0|TRCSEQEVR0}}, x28            // encoding: [0x9c,0x00,0x11,0xd5]
// CHECK: msr      {{trcseqevr1|TRCSEQEVR1}}, x21            // encoding: [0x95,0x01,0x11,0xd5]
// CHECK: msr      {{trcseqevr2|TRCSEQEVR2}}, x16            // encoding: [0x90,0x02,0x11,0xd5]
// CHECK: msr      {{trcseqrstevr|TRCSEQRSTEVR}}, x16          // encoding: [0x90,0x06,0x11,0xd5]
// CHECK: msr      {{trcseqstr|TRCSEQSTR}}, x25             // encoding: [0x99,0x07,0x11,0xd5]
// CHECK: msr      {{trcextinselr|TRCEXTINSELR}}, x29         // encoding: [0x9d,0x08,0x11,0xd5]
// CHECK: msr      {{trccntrldvr0|TRCCNTRLDVR0}}, x20          // encoding: [0xb4,0x00,0x11,0xd5]
// CHECK: msr      {{trccntrldvr1|TRCCNTRLDVR1}}, x20          // encoding: [0xb4,0x01,0x11,0xd5]
// CHECK: msr      {{trccntrldvr2|TRCCNTRLDVR2}}, x22          // encoding: [0xb6,0x02,0x11,0xd5]
// CHECK: msr      {{trccntrldvr3|TRCCNTRLDVR3}}, x12          // encoding: [0xac,0x03,0x11,0xd5]
// CHECK: msr      {{trccntctlr0|TRCCNTCTLR0}}, x20           // encoding: [0xb4,0x04,0x11,0xd5]
// CHECK: msr      {{trccntctlr1|TRCCNTCTLR1}}, x4            // encoding: [0xa4,0x05,0x11,0xd5]
// CHECK: msr      {{trccntctlr2|TRCCNTCTLR2}}, x8            // encoding: [0xa8,0x06,0x11,0xd5]
// CHECK: msr      {{trccntctlr3|TRCCNTCTLR3}}, x16           // encoding: [0xb0,0x07,0x11,0xd5]
// CHECK: msr      {{trccntvr0|TRCCNTVR0}}, x5              // encoding: [0xa5,0x08,0x11,0xd5]
// CHECK: msr      {{trccntvr1|TRCCNTVR1}}, x27             // encoding: [0xbb,0x09,0x11,0xd5]
// CHECK: msr      {{trccntvr2|TRCCNTVR2}}, x21             // encoding: [0xb5,0x0a,0x11,0xd5]
// CHECK: msr      {{trccntvr3|TRCCNTVR3}}, x8              // encoding: [0xa8,0x0b,0x11,0xd5]
// CHECK: msr      {{trcimspec0|TRCIMSPEC0}}, x6             // encoding: [0xe6,0x00,0x11,0xd5]
// CHECK: msr      {{trcimspec1|TRCIMSPEC1}}, x27            // encoding: [0xfb,0x01,0x11,0xd5]
// CHECK: msr      {{trcimspec2|TRCIMSPEC2}}, x23            // encoding: [0xf7,0x02,0x11,0xd5]
// CHECK: msr      {{trcimspec3|TRCIMSPEC3}}, x15            // encoding: [0xef,0x03,0x11,0xd5]
// CHECK: msr      {{trcimspec4|TRCIMSPEC4}}, x13            // encoding: [0xed,0x04,0x11,0xd5]
// CHECK: msr      {{trcimspec5|TRCIMSPEC5}}, x25            // encoding: [0xf9,0x05,0x11,0xd5]
// CHECK: msr      {{trcimspec6|TRCIMSPEC6}}, x19            // encoding: [0xf3,0x06,0x11,0xd5]
// CHECK: msr      {{trcimspec7|TRCIMSPEC7}}, x27            // encoding: [0xfb,0x07,0x11,0xd5]
// CHECK: msr      {{trcrsctlr2|TRCRSCTLR2}}, x4             // encoding: [0x04,0x12,0x11,0xd5]
// CHECK: msr      {{trcrsctlr3|TRCRSCTLR3}}, x0             // encoding: [0x00,0x13,0x11,0xd5]
// CHECK: msr      {{trcrsctlr4|TRCRSCTLR4}}, x21            // encoding: [0x15,0x14,0x11,0xd5]
// CHECK: msr      {{trcrsctlr5|TRCRSCTLR5}}, x8             // encoding: [0x08,0x15,0x11,0xd5]
// CHECK: msr      {{trcrsctlr6|TRCRSCTLR6}}, x20            // encoding: [0x14,0x16,0x11,0xd5]
// CHECK: msr      {{trcrsctlr7|TRCRSCTLR7}}, x11            // encoding: [0x0b,0x17,0x11,0xd5]
// CHECK: msr      {{trcrsctlr8|TRCRSCTLR8}}, x18            // encoding: [0x12,0x18,0x11,0xd5]
// CHECK: msr      {{trcrsctlr9|TRCRSCTLR9}}, x24            // encoding: [0x18,0x19,0x11,0xd5]
// CHECK: msr      {{trcrsctlr10|TRCRSCTLR10}}, x15           // encoding: [0x0f,0x1a,0x11,0xd5]
// CHECK: msr      {{trcrsctlr11|TRCRSCTLR11}}, x21           // encoding: [0x15,0x1b,0x11,0xd5]
// CHECK: msr      {{trcrsctlr12|TRCRSCTLR12}}, x4            // encoding: [0x04,0x1c,0x11,0xd5]
// CHECK: msr      {{trcrsctlr13|TRCRSCTLR13}}, x28           // encoding: [0x1c,0x1d,0x11,0xd5]
// CHECK: msr      {{trcrsctlr14|TRCRSCTLR14}}, x3            // encoding: [0x03,0x1e,0x11,0xd5]
// CHECK: msr      {{trcrsctlr15|TRCRSCTLR15}}, x20           // encoding: [0x14,0x1f,0x11,0xd5]
// CHECK: msr      {{trcrsctlr16|TRCRSCTLR16}}, x12           // encoding: [0x2c,0x10,0x11,0xd5]
// CHECK: msr      {{trcrsctlr17|TRCRSCTLR17}}, x17           // encoding: [0x31,0x11,0x11,0xd5]
// CHECK: msr      {{trcrsctlr18|TRCRSCTLR18}}, x10           // encoding: [0x2a,0x12,0x11,0xd5]
// CHECK: msr      {{trcrsctlr19|TRCRSCTLR19}}, x11           // encoding: [0x2b,0x13,0x11,0xd5]
// CHECK: msr      {{trcrsctlr20|TRCRSCTLR20}}, x3            // encoding: [0x23,0x14,0x11,0xd5]
// CHECK: msr      {{trcrsctlr21|TRCRSCTLR21}}, x18           // encoding: [0x32,0x15,0x11,0xd5]
// CHECK: msr      {{trcrsctlr22|TRCRSCTLR22}}, x26           // encoding: [0x3a,0x16,0x11,0xd5]
// CHECK: msr      {{trcrsctlr23|TRCRSCTLR23}}, x5            // encoding: [0x25,0x17,0x11,0xd5]
// CHECK: msr      {{trcrsctlr24|TRCRSCTLR24}}, x25           // encoding: [0x39,0x18,0x11,0xd5]
// CHECK: msr      {{trcrsctlr25|TRCRSCTLR25}}, x5            // encoding: [0x25,0x19,0x11,0xd5]
// CHECK: msr      {{trcrsctlr26|TRCRSCTLR26}}, x4            // encoding: [0x24,0x1a,0x11,0xd5]
// CHECK: msr      {{trcrsctlr27|TRCRSCTLR27}}, x20           // encoding: [0x34,0x1b,0x11,0xd5]
// CHECK: msr      {{trcrsctlr28|TRCRSCTLR28}}, x5            // encoding: [0x25,0x1c,0x11,0xd5]
// CHECK: msr      {{trcrsctlr29|TRCRSCTLR29}}, x10           // encoding: [0x2a,0x1d,0x11,0xd5]
// CHECK: msr      {{trcrsctlr30|TRCRSCTLR30}}, x24           // encoding: [0x38,0x1e,0x11,0xd5]
// CHECK: msr      {{trcrsctlr31|TRCRSCTLR31}}, x20           // encoding: [0x34,0x1f,0x11,0xd5]
// CHECK: msr      {{trcssccr0|TRCSSCCR0}}, x23             // encoding: [0x57,0x10,0x11,0xd5]
// CHECK: msr      {{trcssccr1|TRCSSCCR1}}, x27             // encoding: [0x5b,0x11,0x11,0xd5]
// CHECK: msr      {{trcssccr2|TRCSSCCR2}}, x27             // encoding: [0x5b,0x12,0x11,0xd5]
// CHECK: msr      {{trcssccr3|TRCSSCCR3}}, x6              // encoding: [0x46,0x13,0x11,0xd5]
// CHECK: msr      {{trcssccr4|TRCSSCCR4}}, x3              // encoding: [0x43,0x14,0x11,0xd5]
// CHECK: msr      {{trcssccr5|TRCSSCCR5}}, x12             // encoding: [0x4c,0x15,0x11,0xd5]
// CHECK: msr      {{trcssccr6|TRCSSCCR6}}, x7              // encoding: [0x47,0x16,0x11,0xd5]
// CHECK: msr      {{trcssccr7|TRCSSCCR7}}, x6              // encoding: [0x46,0x17,0x11,0xd5]
// CHECK: msr      {{trcsscsr0|TRCSSCSR0}}, x20             // encoding: [0x54,0x18,0x11,0xd5]
// CHECK: msr      {{trcsscsr1|TRCSSCSR1}}, x17             // encoding: [0x51,0x19,0x11,0xd5]
// CHECK: msr      {{trcsscsr2|TRCSSCSR2}}, x11             // encoding: [0x4b,0x1a,0x11,0xd5]
// CHECK: msr      {{trcsscsr3|TRCSSCSR3}}, x4              // encoding: [0x44,0x1b,0x11,0xd5]
// CHECK: msr      {{trcsscsr4|TRCSSCSR4}}, x14             // encoding: [0x4e,0x1c,0x11,0xd5]
// CHECK: msr      {{trcsscsr5|TRCSSCSR5}}, x22             // encoding: [0x56,0x1d,0x11,0xd5]
// CHECK: msr      {{trcsscsr6|TRCSSCSR6}}, x3              // encoding: [0x43,0x1e,0x11,0xd5]
// CHECK: msr      {{trcsscsr7|TRCSSCSR7}}, x11             // encoding: [0x4b,0x1f,0x11,0xd5]
// CHECK: msr      {{trcsspcicr0|TRCSSPCICR0}}, x2            // encoding: [0x62,0x10,0x11,0xd5]
// CHECK: msr      {{trcsspcicr1|TRCSSPCICR1}}, x3            // encoding: [0x63,0x11,0x11,0xd5]
// CHECK: msr      {{trcsspcicr2|TRCSSPCICR2}}, x5            // encoding: [0x65,0x12,0x11,0xd5]
// CHECK: msr      {{trcsspcicr3|TRCSSPCICR3}}, x7            // encoding: [0x67,0x13,0x11,0xd5]
// CHECK: msr      {{trcsspcicr4|TRCSSPCICR4}}, x11           // encoding: [0x6b,0x14,0x11,0xd5]
// CHECK: msr      {{trcsspcicr5|TRCSSPCICR5}}, x13           // encoding: [0x6d,0x15,0x11,0xd5]
// CHECK: msr      {{trcsspcicr6|TRCSSPCICR6}}, x17           // encoding: [0x71,0x16,0x11,0xd5]
// CHECK: msr      {{trcsspcicr7|TRCSSPCICR7}}, x23           // encoding: [0x77,0x17,0x11,0xd5]
// CHECK: msr      {{trcpdcr|TRCPDCR}}, x3                // encoding: [0x83,0x14,0x11,0xd5]
// CHECK: msr      {{trcacvr0|TRCACVR0}}, x6               // encoding: [0x06,0x20,0x11,0xd5]
// CHECK: msr      {{trcacvr1|TRCACVR1}}, x20              // encoding: [0x14,0x22,0x11,0xd5]
// CHECK: msr      {{trcacvr2|TRCACVR2}}, x25              // encoding: [0x19,0x24,0x11,0xd5]
// CHECK: msr      {{trcacvr3|TRCACVR3}}, x1               // encoding: [0x01,0x26,0x11,0xd5]
// CHECK: msr      {{trcacvr4|TRCACVR4}}, x28              // encoding: [0x1c,0x28,0x11,0xd5]
// CHECK: msr      {{trcacvr5|TRCACVR5}}, x15              // encoding: [0x0f,0x2a,0x11,0xd5]
// CHECK: msr      {{trcacvr6|TRCACVR6}}, x25              // encoding: [0x19,0x2c,0x11,0xd5]
// CHECK: msr      {{trcacvr7|TRCACVR7}}, x12              // encoding: [0x0c,0x2e,0x11,0xd5]
// CHECK: msr      {{trcacvr8|TRCACVR8}}, x5               // encoding: [0x25,0x20,0x11,0xd5]
// CHECK: msr      {{trcacvr9|TRCACVR9}}, x25              // encoding: [0x39,0x22,0x11,0xd5]
// CHECK: msr      {{trcacvr10|TRCACVR10}}, x13             // encoding: [0x2d,0x24,0x11,0xd5]
// CHECK: msr      {{trcacvr11|TRCACVR11}}, x10             // encoding: [0x2a,0x26,0x11,0xd5]
// CHECK: msr      {{trcacvr12|TRCACVR12}}, x19             // encoding: [0x33,0x28,0x11,0xd5]
// CHECK: msr      {{trcacvr13|TRCACVR13}}, x10             // encoding: [0x2a,0x2a,0x11,0xd5]
// CHECK: msr      {{trcacvr14|TRCACVR14}}, x19             // encoding: [0x33,0x2c,0x11,0xd5]
// CHECK: msr      {{trcacvr15|TRCACVR15}}, x2              // encoding: [0x22,0x2e,0x11,0xd5]
// CHECK: msr      {{trcacatr0|TRCACATR0}}, x15             // encoding: [0x4f,0x20,0x11,0xd5]
// CHECK: msr      {{trcacatr1|TRCACATR1}}, x13             // encoding: [0x4d,0x22,0x11,0xd5]
// CHECK: msr      {{trcacatr2|TRCACATR2}}, x8              // encoding: [0x48,0x24,0x11,0xd5]
// CHECK: msr      {{trcacatr3|TRCACATR3}}, x1              // encoding: [0x41,0x26,0x11,0xd5]
// CHECK: msr      {{trcacatr4|TRCACATR4}}, x11             // encoding: [0x4b,0x28,0x11,0xd5]
// CHECK: msr      {{trcacatr5|TRCACATR5}}, x8              // encoding: [0x48,0x2a,0x11,0xd5]
// CHECK: msr      {{trcacatr6|TRCACATR6}}, x24             // encoding: [0x58,0x2c,0x11,0xd5]
// CHECK: msr      {{trcacatr7|TRCACATR7}}, x6              // encoding: [0x46,0x2e,0x11,0xd5]
// CHECK: msr      {{trcacatr8|TRCACATR8}}, x23             // encoding: [0x77,0x20,0x11,0xd5]
// CHECK: msr      {{trcacatr9|TRCACATR9}}, x5              // encoding: [0x65,0x22,0x11,0xd5]
// CHECK: msr      {{trcacatr10|TRCACATR10}}, x11            // encoding: [0x6b,0x24,0x11,0xd5]
// CHECK: msr      {{trcacatr11|TRCACATR11}}, x11            // encoding: [0x6b,0x26,0x11,0xd5]
// CHECK: msr      {{trcacatr12|TRCACATR12}}, x3             // encoding: [0x63,0x28,0x11,0xd5]
// CHECK: msr      {{trcacatr13|TRCACATR13}}, x28            // encoding: [0x7c,0x2a,0x11,0xd5]
// CHECK: msr      {{trcacatr14|TRCACATR14}}, x25            // encoding: [0x79,0x2c,0x11,0xd5]
// CHECK: msr      {{trcacatr15|TRCACATR15}}, x4             // encoding: [0x64,0x2e,0x11,0xd5]
// CHECK: msr      {{trcdvcvr0|TRCDVCVR0}}, x6              // encoding: [0x86,0x20,0x11,0xd5]
// CHECK: msr      {{trcdvcvr1|TRCDVCVR1}}, x3              // encoding: [0x83,0x24,0x11,0xd5]
// CHECK: msr      {{trcdvcvr2|TRCDVCVR2}}, x5              // encoding: [0x85,0x28,0x11,0xd5]
// CHECK: msr      {{trcdvcvr3|TRCDVCVR3}}, x11             // encoding: [0x8b,0x2c,0x11,0xd5]
// CHECK: msr      {{trcdvcvr4|TRCDVCVR4}}, x9              // encoding: [0xa9,0x20,0x11,0xd5]
// CHECK: msr      {{trcdvcvr5|TRCDVCVR5}}, x14             // encoding: [0xae,0x24,0x11,0xd5]
// CHECK: msr      {{trcdvcvr6|TRCDVCVR6}}, x10             // encoding: [0xaa,0x28,0x11,0xd5]
// CHECK: msr      {{trcdvcvr7|TRCDVCVR7}}, x12             // encoding: [0xac,0x2c,0x11,0xd5]
// CHECK: msr      {{trcdvcmr0|TRCDVCMR0}}, x8              // encoding: [0xc8,0x20,0x11,0xd5]
// CHECK: msr      {{trcdvcmr1|TRCDVCMR1}}, x8              // encoding: [0xc8,0x24,0x11,0xd5]
// CHECK: msr      {{trcdvcmr2|TRCDVCMR2}}, x22             // encoding: [0xd6,0x28,0x11,0xd5]
// CHECK: msr      {{trcdvcmr3|TRCDVCMR3}}, x22             // encoding: [0xd6,0x2c,0x11,0xd5]
// CHECK: msr      {{trcdvcmr4|TRCDVCMR4}}, x5              // encoding: [0xe5,0x20,0x11,0xd5]
// CHECK: msr      {{trcdvcmr5|TRCDVCMR5}}, x16             // encoding: [0xf0,0x24,0x11,0xd5]
// CHECK: msr      {{trcdvcmr6|TRCDVCMR6}}, x27             // encoding: [0xfb,0x28,0x11,0xd5]
// CHECK: msr      {{trcdvcmr7|TRCDVCMR7}}, x21             // encoding: [0xf5,0x2c,0x11,0xd5]
// CHECK: msr      {{trccidcvr0|TRCCIDCVR0}}, x8             // encoding: [0x08,0x30,0x11,0xd5]
// CHECK: msr      {{trccidcvr1|TRCCIDCVR1}}, x6             // encoding: [0x06,0x32,0x11,0xd5]
// CHECK: msr      {{trccidcvr2|TRCCIDCVR2}}, x9             // encoding: [0x09,0x34,0x11,0xd5]
// CHECK: msr      {{trccidcvr3|TRCCIDCVR3}}, x8             // encoding: [0x08,0x36,0x11,0xd5]
// CHECK: msr      {{trccidcvr4|TRCCIDCVR4}}, x3             // encoding: [0x03,0x38,0x11,0xd5]
// CHECK: msr      {{trccidcvr5|TRCCIDCVR5}}, x21            // encoding: [0x15,0x3a,0x11,0xd5]
// CHECK: msr      {{trccidcvr6|TRCCIDCVR6}}, x12            // encoding: [0x0c,0x3c,0x11,0xd5]
// CHECK: msr      {{trccidcvr7|TRCCIDCVR7}}, x7             // encoding: [0x07,0x3e,0x11,0xd5]
// CHECK: msr      {{trcvmidcvr0|TRCVMIDCVR0}}, x4            // encoding: [0x24,0x30,0x11,0xd5]
// CHECK: msr      {{trcvmidcvr1|TRCVMIDCVR1}}, x3            // encoding: [0x23,0x32,0x11,0xd5]
// CHECK: msr      {{trcvmidcvr2|TRCVMIDCVR2}}, x9            // encoding: [0x29,0x34,0x11,0xd5]
// CHECK: msr      {{trcvmidcvr3|TRCVMIDCVR3}}, x17           // encoding: [0x31,0x36,0x11,0xd5]
// CHECK: msr      {{trcvmidcvr4|TRCVMIDCVR4}}, x14           // encoding: [0x2e,0x38,0x11,0xd5]
// CHECK: msr      {{trcvmidcvr5|TRCVMIDCVR5}}, x12           // encoding: [0x2c,0x3a,0x11,0xd5]
// CHECK: msr      {{trcvmidcvr6|TRCVMIDCVR6}}, x10           // encoding: [0x2a,0x3c,0x11,0xd5]
// CHECK: msr      {{trcvmidcvr7|TRCVMIDCVR7}}, x3            // encoding: [0x23,0x3e,0x11,0xd5]
// CHECK: msr      {{trccidcctlr0|TRCCIDCCTLR0}}, x14          // encoding: [0x4e,0x30,0x11,0xd5]
// CHECK: msr      {{trccidcctlr1|TRCCIDCCTLR1}}, x22          // encoding: [0x56,0x31,0x11,0xd5]
// CHECK: msr      {{trcvmidcctlr0|TRCVMIDCCTLR0}}, x8          // encoding: [0x48,0x32,0x11,0xd5]
// CHECK: msr      {{trcvmidcctlr1|TRCVMIDCCTLR1}}, x15         // encoding: [0x4f,0x33,0x11,0xd5]
// CHECK: msr      {{trcitctrl|TRCITCTRL}}, x1              // encoding: [0x81,0x70,0x11,0xd5]
// CHECK: msr      {{trcclaimset|TRCCLAIMSET}}, x7            // encoding: [0xc7,0x78,0x11,0xd5]
// CHECK: msr      {{trcclaimclr|TRCCLAIMCLR}}, x29           // encoding: [0xdd,0x79,0x11,0xd5]
