@ RUN: llvm-mc -triple armv7 -mattr=virtualization -show-encoding %s | FileCheck %s --check-prefix=CHECK-ARM
@ RUN: llvm-mc -triple thumbv7 -mattr=virtualization -show-encoding %s | FileCheck %s --check-prefix=CHECK-THUMB

        mrs r2, r8_usr
        mrs r3, r9_usr
        mrs r5, r10_usr
        mrs r7, r11_usr
        mrs r11, r12_usr
        mrs r1, sp_usr
        mrs r2, lr_usr
@ CHECK-ARM:         mrs     r2, r8_usr              @ encoding: [0x00,0x22,0x00,0xe1]
@ CHECK-ARM:         mrs     r3, r9_usr              @ encoding: [0x00,0x32,0x01,0xe1]
@ CHECK-ARM:         mrs     r5, r10_usr             @ encoding: [0x00,0x52,0x02,0xe1]
@ CHECK-ARM:         mrs     r7, r11_usr             @ encoding: [0x00,0x72,0x03,0xe1]
@ CHECK-ARM:         mrs     r11, r12_usr            @ encoding: [0x00,0xb2,0x04,0xe1]
@ CHECK-ARM:         mrs     r1, sp_usr              @ encoding: [0x00,0x12,0x05,0xe1]
@ CHECK-ARM:         mrs     r2, lr_usr              @ encoding: [0x00,0x22,0x06,0xe1]
@ CHECK-THUMB:         mrs     r2, r8_usr              @ encoding: [0xe0,0xf3,0x20,0x82]
@ CHECK-THUMB:         mrs     r3, r9_usr              @ encoding: [0xe1,0xf3,0x20,0x83]
@ CHECK-THUMB:         mrs     r5, r10_usr             @ encoding: [0xe2,0xf3,0x20,0x85]
@ CHECK-THUMB:         mrs     r7, r11_usr             @ encoding: [0xe3,0xf3,0x20,0x87]
@ CHECK-THUMB:         mrs     r11, r12_usr            @ encoding: [0xe4,0xf3,0x20,0x8b]
@ CHECK-THUMB:         mrs     r1, sp_usr              @ encoding: [0xe5,0xf3,0x20,0x81]
@ CHECK-THUMB:         mrs     r2, lr_usr              @ encoding: [0xe6,0xf3,0x20,0x82]

        mrs r2, r8_fiq
        mrs r3, r9_fiq
        mrs r5, r10_fiq
        mrs r7, r11_fiq
        mrs r11, r12_fiq
        mrs r1, sp_fiq
        mrs r2, lr_fiq
        mrs r3, spsr_fiq
@ CHECK-ARM:         mrs     r2, r8_fiq              @ encoding: [0x00,0x22,0x08,0xe1]
@ CHECK-ARM:         mrs     r3, r9_fiq              @ encoding: [0x00,0x32,0x09,0xe1]
@ CHECK-ARM:         mrs     r5, r10_fiq             @ encoding: [0x00,0x52,0x0a,0xe1]
@ CHECK-ARM:         mrs     r7, r11_fiq             @ encoding: [0x00,0x72,0x0b,0xe1]
@ CHECK-ARM:         mrs     r11, r12_fiq            @ encoding: [0x00,0xb2,0x0c,0xe1]
@ CHECK-ARM:         mrs     r1, sp_fiq              @ encoding: [0x00,0x12,0x0d,0xe1]
@ CHECK-ARM:         mrs     r2, lr_fiq              @ encoding: [0x00,0x22,0x0e,0xe1]
@ CHECK-ARM:         mrs     r3, SPSR_fiq            @ encoding: [0x00,0x32,0x4e,0xe1]
@ CHECK-THUMB:         mrs     r2, r8_fiq              @ encoding: [0xe8,0xf3,0x20,0x82]
@ CHECK-THUMB:         mrs     r3, r9_fiq              @ encoding: [0xe9,0xf3,0x20,0x83]
@ CHECK-THUMB:         mrs     r5, r10_fiq             @ encoding: [0xea,0xf3,0x20,0x85]
@ CHECK-THUMB:         mrs     r7, r11_fiq             @ encoding: [0xeb,0xf3,0x20,0x87]
@ CHECK-THUMB:         mrs     r11, r12_fiq            @ encoding: [0xec,0xf3,0x20,0x8b]
@ CHECK-THUMB:         mrs     r1, sp_fiq              @ encoding: [0xed,0xf3,0x20,0x81]
@ CHECK-THUMB:         mrs     r2, lr_fiq              @ encoding: [0xee,0xf3,0x20,0x82]
@ CHECK-THUMB:         mrs     r3, SPSR_fiq            @ encoding: [0xfe,0xf3,0x20,0x83]

        mrs r4, lr_irq
        mrs r9, sp_irq
        mrs r1, spsr_irq
@ CHECK-ARM:         mrs     r4, lr_irq              @ encoding: [0x00,0x43,0x00,0xe1]
@ CHECK-ARM:         mrs     r9, sp_irq              @ encoding: [0x00,0x93,0x01,0xe1]
@ CHECK-ARM:         mrs     r1, SPSR_irq            @ encoding: [0x00,0x13,0x40,0xe1]
@ CHECK-THUMB:         mrs     r4, lr_irq              @ encoding: [0xe0,0xf3,0x30,0x84]
@ CHECK-THUMB:         mrs     r9, sp_irq              @ encoding: [0xe1,0xf3,0x30,0x89]
@ CHECK-THUMB:         mrs     r1, SPSR_irq            @ encoding: [0xf0,0xf3,0x30,0x81]

        mrs r1, lr_svc
        mrs r3, sp_svc
        mrs r5, spsr_svc
@ CHECK-ARM:         mrs     r1, lr_svc              @ encoding: [0x00,0x13,0x02,0xe1]
@ CHECK-ARM:         mrs     r3, sp_svc              @ encoding: [0x00,0x33,0x03,0xe1]
@ CHECK-ARM:         mrs     r5, SPSR_svc            @ encoding: [0x00,0x53,0x42,0xe1]
@ CHECK-THUMB:         mrs     r1, lr_svc              @ encoding: [0xe2,0xf3,0x30,0x81]
@ CHECK-THUMB:         mrs     r3, sp_svc              @ encoding: [0xe3,0xf3,0x30,0x83]
@ CHECK-THUMB:         mrs     r5, SPSR_svc            @ encoding: [0xf2,0xf3,0x30,0x85]

        mrs r5, lr_abt
        mrs r7, sp_abt
        mrs r9, spsr_abt
@ CHECK-ARM:         mrs     r5, lr_abt              @ encoding: [0x00,0x53,0x04,0xe1]
@ CHECK-ARM:         mrs     r7, sp_abt              @ encoding: [0x00,0x73,0x05,0xe1]
@ CHECK-ARM:         mrs     r9, SPSR_abt            @ encoding: [0x00,0x93,0x44,0xe1]
@ CHECK-THUMB:         mrs     r5, lr_abt              @ encoding: [0xe4,0xf3,0x30,0x85]
@ CHECK-THUMB:         mrs     r7, sp_abt              @ encoding: [0xe5,0xf3,0x30,0x87]
@ CHECK-THUMB:         mrs     r9, SPSR_abt            @ encoding: [0xf4,0xf3,0x30,0x89]

        mrs r9, lr_und
        mrs r11, sp_und
        mrs r12, spsr_und
@ CHECK-ARM:         mrs     r9, lr_und              @ encoding: [0x00,0x93,0x06,0xe1]
@ CHECK-ARM:         mrs     r11, sp_und             @ encoding: [0x00,0xb3,0x07,0xe1]
@ CHECK-ARM:         mrs     r12, SPSR_und           @ encoding: [0x00,0xc3,0x46,0xe1]
@ CHECK-THUMB:         mrs     r9, lr_und              @ encoding: [0xe6,0xf3,0x30,0x89]
@ CHECK-THUMB:         mrs     r11, sp_und             @ encoding: [0xe7,0xf3,0x30,0x8b]
@ CHECK-THUMB:         mrs     r12, SPSR_und           @ encoding: [0xf6,0xf3,0x30,0x8c]


        mrs r2, lr_mon
        mrs r4, sp_mon
        mrs r6, spsr_mon
@ CHECK-ARM:         mrs     r2, lr_mon              @ encoding: [0x00,0x23,0x0c,0xe1]
@ CHECK-ARM:         mrs     r4, sp_mon              @ encoding: [0x00,0x43,0x0d,0xe1]
@ CHECK-ARM:         mrs     r6, SPSR_mon            @ encoding: [0x00,0x63,0x4c,0xe1]
@ CHECK-THUMB:         mrs     r2, lr_mon              @ encoding: [0xec,0xf3,0x30,0x82]
@ CHECK-THUMB:         mrs     r4, sp_mon              @ encoding: [0xed,0xf3,0x30,0x84]
@ CHECK-THUMB:         mrs     r6, SPSR_mon            @ encoding: [0xfc,0xf3,0x30,0x86]


        mrs r6, elr_hyp
        mrs r8, sp_hyp
        mrs r10, spsr_hyp
@ CHECK-ARM:         mrs     r6, elr_hyp             @ encoding: [0x00,0x63,0x0e,0xe1]
@ CHECK-ARM:         mrs     r8, sp_hyp              @ encoding: [0x00,0x83,0x0f,0xe1]
@ CHECK-ARM:         mrs     r10, SPSR_hyp            @ encoding: [0x00,0xa3,0x4e,0xe1]
@ CHECK-THUMB:         mrs     r6, elr_hyp             @ encoding: [0xee,0xf3,0x30,0x86]
@ CHECK-THUMB:         mrs     r8, sp_hyp              @ encoding: [0xef,0xf3,0x30,0x88]
@ CHECK-THUMB:         mrs     r10, SPSR_hyp            @ encoding: [0xfe,0xf3,0x30,0x8a]


        msr r8_usr, r2
        msr r9_usr, r3
        msr r10_usr, r5
        msr r11_usr, r7
        msr r12_usr, r11
        msr sp_usr, r1
        msr lr_usr, r2
@ CHECK-ARM:         msr     r8_usr, r2              @ encoding: [0x02,0xf2,0x20,0xe1]
@ CHECK-ARM:         msr     r9_usr, r3              @ encoding: [0x03,0xf2,0x21,0xe1]
@ CHECK-ARM:         msr     r10_usr, r5             @ encoding: [0x05,0xf2,0x22,0xe1]
@ CHECK-ARM:         msr     r11_usr, r7             @ encoding: [0x07,0xf2,0x23,0xe1]
@ CHECK-ARM:         msr     r12_usr, r11            @ encoding: [0x0b,0xf2,0x24,0xe1]
@ CHECK-ARM:         msr     sp_usr, r1              @ encoding: [0x01,0xf2,0x25,0xe1]
@ CHECK-ARM:         msr     lr_usr, r2              @ encoding: [0x02,0xf2,0x26,0xe1]
@ CHECK-THUMB:         msr     r8_usr, r2              @ encoding: [0x82,0xf3,0x20,0x80]
@ CHECK-THUMB:         msr     r9_usr, r3              @ encoding: [0x83,0xf3,0x20,0x81]
@ CHECK-THUMB:         msr     r10_usr, r5             @ encoding: [0x85,0xf3,0x20,0x82]
@ CHECK-THUMB:         msr     r11_usr, r7             @ encoding: [0x87,0xf3,0x20,0x83]
@ CHECK-THUMB:         msr     r12_usr, r11            @ encoding: [0x8b,0xf3,0x20,0x84]
@ CHECK-THUMB:         msr     sp_usr, r1              @ encoding: [0x81,0xf3,0x20,0x85]
@ CHECK-THUMB:         msr     lr_usr, r2              @ encoding: [0x82,0xf3,0x20,0x86]

        msr r8_fiq, r2
        msr r9_fiq, r3
        msr r10_fiq, r5
        msr r11_fiq, r7
        msr r12_fiq, r11
        msr sp_fiq, r1
        msr lr_fiq, r2
        msr spsr_fiq, r3
@ CHECK-ARM:         msr     r8_fiq, r2              @ encoding: [0x02,0xf2,0x28,0xe1]
@ CHECK-ARM:         msr     r9_fiq, r3              @ encoding: [0x03,0xf2,0x29,0xe1]
@ CHECK-ARM:         msr     r10_fiq, r5             @ encoding: [0x05,0xf2,0x2a,0xe1]
@ CHECK-ARM:         msr     r11_fiq, r7             @ encoding: [0x07,0xf2,0x2b,0xe1]
@ CHECK-ARM:         msr     r12_fiq, r11            @ encoding: [0x0b,0xf2,0x2c,0xe1]
@ CHECK-ARM:         msr     sp_fiq, r1              @ encoding: [0x01,0xf2,0x2d,0xe1]
@ CHECK-ARM:         msr     lr_fiq, r2              @ encoding: [0x02,0xf2,0x2e,0xe1]
@ CHECK-ARM:         msr     SPSR_fiq, r3            @ encoding: [0x03,0xf2,0x6e,0xe1]
@ CHECK-THUMB:         msr     r8_fiq, r2              @ encoding: [0x82,0xf3,0x20,0x88]
@ CHECK-THUMB:         msr     r9_fiq, r3              @ encoding: [0x83,0xf3,0x20,0x89]
@ CHECK-THUMB:         msr     r10_fiq, r5             @ encoding: [0x85,0xf3,0x20,0x8a]
@ CHECK-THUMB:         msr     r11_fiq, r7             @ encoding: [0x87,0xf3,0x20,0x8b]
@ CHECK-THUMB:         msr     r12_fiq, r11            @ encoding: [0x8b,0xf3,0x20,0x8c]
@ CHECK-THUMB:         msr     sp_fiq, r1              @ encoding: [0x81,0xf3,0x20,0x8d]
@ CHECK-THUMB:         msr     lr_fiq, r2              @ encoding: [0x82,0xf3,0x20,0x8e]
@ CHECK-THUMB:        msr     SPSR_fiq, r3            @ encoding: [0x93,0xf3,0x20,0x8e]

        msr lr_irq, r4
        msr sp_irq, r9
        msr spsr_irq, r11
@ CHECK-ARM:         msr     lr_irq, r4              @ encoding: [0x04,0xf3,0x20,0xe1]
@ CHECK-ARM:         msr     sp_irq, r9              @ encoding: [0x09,0xf3,0x21,0xe1]
@ CHECK-ARM:         msr     SPSR_irq, r11           @ encoding: [0x0b,0xf3,0x60,0xe1]
@ CHECK-THUMB:         msr     lr_irq, r4              @ encoding: [0x84,0xf3,0x30,0x80]
@ CHECK-THUMB:         msr     sp_irq, r9              @ encoding: [0x89,0xf3,0x30,0x81]
@ CHECK-THUMB:         msr     SPSR_irq, r11           @ encoding: [0x9b,0xf3,0x30,0x80]

        msr lr_svc, r1
        msr sp_svc, r3
        msr spsr_svc, r5
@ CHECK-ARM:         msr     lr_svc, r1              @ encoding: [0x01,0xf3,0x22,0xe1]
@ CHECK-ARM:         msr     sp_svc, r3              @ encoding: [0x03,0xf3,0x23,0xe1]
@ CHECK-ARM:         msr     SPSR_svc, r5            @ encoding: [0x05,0xf3,0x62,0xe1]
@ CHECK-THUMB:         msr     lr_svc, r1              @ encoding: [0x81,0xf3,0x30,0x82]
@ CHECK-THUMB:         msr     sp_svc, r3              @ encoding: [0x83,0xf3,0x30,0x83]
@ CHECK-THUMB:         msr     SPSR_svc, r5            @ encoding: [0x95,0xf3,0x30,0x82]

        msr lr_abt, r5
        msr sp_abt, r7
        msr spsr_abt, r9
@ CHECK-ARM:         msr     lr_abt, r5              @ encoding: [0x05,0xf3,0x24,0xe1]
@ CHECK-ARM:         msr     sp_abt, r7              @ encoding: [0x07,0xf3,0x25,0xe1]
@ CHECK-ARM:         msr     SPSR_abt, r9            @ encoding: [0x09,0xf3,0x64,0xe1]
@ CHECK-THUMB:         msr     lr_abt, r5              @ encoding: [0x85,0xf3,0x30,0x84]
@ CHECK-THUMB:         msr     sp_abt, r7              @ encoding: [0x87,0xf3,0x30,0x85]
@ CHECK-THUMB:         msr     SPSR_abt, r9            @ encoding: [0x99,0xf3,0x30,0x84]

        msr lr_und, r9
        msr sp_und, r11
        msr spsr_und, r12
@ CHECK-ARM:         msr     lr_und, r9              @ encoding: [0x09,0xf3,0x26,0xe1]
@ CHECK-ARM:         msr     sp_und, r11             @ encoding: [0x0b,0xf3,0x27,0xe1]
@ CHECK-ARM:         msr     SPSR_und, r12           @ encoding: [0x0c,0xf3,0x66,0xe1]
@ CHECK-THUMB:         msr     lr_und, r9              @ encoding: [0x89,0xf3,0x30,0x86]
@ CHECK-THUMB:         msr     sp_und, r11             @ encoding: [0x8b,0xf3,0x30,0x87]
@ CHECK-THUMB:         msr     SPSR_und, r12           @ encoding: [0x9c,0xf3,0x30,0x86]


        msr lr_mon, r2
        msr sp_mon, r4
        msr spsr_mon, r6
@ CHECK-ARM:         msr     lr_mon, r2              @ encoding: [0x02,0xf3,0x2c,0xe1]
@ CHECK-ARM:         msr     sp_mon, r4              @ encoding: [0x04,0xf3,0x2d,0xe1]
@ CHECK-ARM:         msr     SPSR_mon, r6            @ encoding: [0x06,0xf3,0x6c,0xe1]
@ CHECK-THUMB:         msr     lr_mon, r2              @ encoding: [0x82,0xf3,0x30,0x8c]
@ CHECK-THUMB:         msr     sp_mon, r4              @ encoding: [0x84,0xf3,0x30,0x8d]
@ CHECK-THUMB:         msr     SPSR_mon, r6            @ encoding: [0x96,0xf3,0x30,0x8c]

        msr elr_hyp, r6
        msr sp_hyp, r8
        msr spsr_hyp, r10
@ CHECK-ARM:         msr     elr_hyp, r6             @ encoding: [0x06,0xf3,0x2e,0xe1]
@ CHECK-ARM:         msr     sp_hyp, r8              @ encoding: [0x08,0xf3,0x2f,0xe1]
@ CHECK-ARM:         msr     SPSR_hyp, r10           @ encoding: [0x0a,0xf3,0x6e,0xe1]
@ CHECK-THUMB:         msr     elr_hyp, r6             @ encoding: [0x86,0xf3,0x30,0x8e]
@ CHECK-THUMB:         msr     sp_hyp, r8              @ encoding: [0x88,0xf3,0x30,0x8f]
@ CHECK-THUMB:         msr     SPSR_hyp, r10           @ encoding: [0x9a,0xf3,0x30,0x8e]
