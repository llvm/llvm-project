@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s

@ fconstd/fconsts aliases
        fconsts s4, #0x0
        fconsts s4, #0x70
        fconstd d3, #0x0
        fconstd d3, #0x70

        fconstsne s5, #0x1
        fconstsgt s5, #0x20
        fconstdlt d2, #0x3
        fconstdge d2, #0x40

@ CHECK: vmov.f32        s4, #2.000000e+00 @ encoding: [0x00,0x2a,0xb0,0xee]
@ CHECK: vmov.f32        s4, #1.000000e+00 @ encoding: [0x00,0x2a,0xb7,0xee]
@ CHECK: vmov.f64        d3, #2.000000e+00 @ encoding: [0x00,0x3b,0xb0,0xee]
@ CHECK: vmov.f64        d3, #1.000000e+00 @ encoding: [0x00,0x3b,0xb7,0xee]

@ CHECK: vmovne.f32      s5, #2.125000e+00 @ encoding: [0x01,0x2a,0xf0,0x1e]
@ CHECK: vmovgt.f32      s5, #8.000000e+00 @ encoding: [0x00,0x2a,0xf2,0xce]
@ CHECK: vmovlt.f64      d2, #2.375000e+00 @ encoding: [0x03,0x2b,0xb0,0xbe]
@ CHECK: vmovge.f64      d2, #1.250000e-01 @ encoding: [0x00,0x2b,0xb4,0xae]
