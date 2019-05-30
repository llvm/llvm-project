// Some simple operations on S, D and Q registers (loads, stores and moves) are
// also avaliable in MVE, even in the integer-only version. Some of these
// instructions (operating on D or Q registers, or FP16 values) are only
// available for certain targets.

// Note that it's not always obvious which instructions are available, for
// example several instructions operating on D registers are available for
// single-precision only FPUs.

// All of these instructions are rejected if no VFP or MVE features are
// present.
// RUN: not llvm-mc -triple=thumbv8.1m.main -show-encoding 2>%t < %s
// RUN: FileCheck %s < %t --check-prefix=NOFP16 --check-prefix=NOFP32 --check-prefix=NOFP64

// VFP and NEON implementations by default have FP32 and FP64, but not FP16.
// The VFPv3 FP16 extension just added conversion instructions, which we don't
// care about here.
// RUN: not llvm-mc -triple=thumbv8.1m.main -show-encoding -mattr=+vfp2 2>%t < %s | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=FP32 --check-prefix=FP64
// RUN: FileCheck %s < %t --check-prefix=NOFP16
// RUN: not llvm-mc -triple=thumbv8.1m.main -show-encoding -mattr=+fp-armv8,+neon 2>%t < %s | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=FP32 --check-prefix=FP64
// RUN: FileCheck %s < %t --check-prefix=NOFP16

// The v8.2A FP16 extension added loads, stores and moves for FP16.
// RUN: llvm-mc -triple=thumbv8.1m.main -show-encoding -mattr=+fp-armv8,+fullfp16 < %s | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=FP16 --check-prefix=FP32 --check-prefix=FP64

// M-profile FPUs (e.g. Cortex-M4/M7/M33) do not have FP16 instructions, and
// the FP64 instructions are optional. They are also limited to 16 D registers,
// but we don't test that here.
// RUN: not llvm-mc -triple=thumbv8.1m.main -show-encoding -mattr=+vfp4d16sp 2>%t < %s | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=FP32
// RUN: FileCheck %s < %t --check-prefix=NOFP16 --check-prefix=NOFP64
// RUN: not llvm-mc -triple=thumbv8.1m.main -show-encoding -mattr=+vfp4,-d32 2>%t < %s | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=FP32 --check-prefix=FP64
// RUN: FileCheck %s < %t --check-prefix=NOFP16

vldmia  r0, {d0}
# FP32: vldmia  r0, {d0}               @ encoding: [0x90,0xec,0x02,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vstmia  r0, {d0}
# FP32: vstmia  r0, {d0}                @ encoding: [0x80,0xec,0x02,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vldmia  r0, {s0}
# FP32: vldmia  r0, {s0}                @ encoding: [0x90,0xec,0x01,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vstmia  r0, {s0}
# FP32: vstmia  r0, {s0}                @ encoding: [0x80,0xec,0x01,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

fldmdbx r0!, {d0}
# FP32: fldmdbx r0!, {d0}               @ encoding: [0x30,0xed,0x03,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

fstmiax r0, {d0}
# FP32: fstmiax r0, {d0}                @ encoding: [0x80,0xec,0x03,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vldr.16 s0, [r0]
# FP16: vldr.16 s0, [r0]                @ encoding: [0x90,0xed,0x00,0x09]
# NOFP16: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: 16-bit fp registers

vldr s0, [r0]
# FP32: vldr    s0, [r0]                @ encoding: [0x90,0xed,0x00,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vldr d0, [r0]
# FP32: vldr    d0, [r0]                @ encoding: [0x90,0xed,0x00,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vstr.16 s0, [r0]
# FP16: vstr.16 s0, [r0]                @ encoding: [0x80,0xed,0x00,0x09]
# NOFP16: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: 16-bit fp registers

vstr s0, [r0]
# FP32: vstr    s0, [r0]                @ encoding: [0x80,0xed,0x00,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vstr d0, [r0]
# FP32: vstr    d0, [r0]                @ encoding: [0x80,0xed,0x00,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vmov.f16 r0, s0
# FP16: vmov.f16        r0, s0          @ encoding: [0x10,0xee,0x10,0x09]
# NOFP16: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: 16-bit fp registers

vmov.f16 s0, r0
# FP16: vmov.f16        s0, r0          @ encoding: [0x00,0xee,0x10,0x09]
# NOFP16: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: 16-bit fp registers

vmov s0, r0
# FP32: vmov    s0, r0                  @ encoding: [0x00,0xee,0x10,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vmov r0, s0
# FP32: vmov    r0, s0                  @ encoding: [0x10,0xee,0x10,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vmov r0, r1, d0
# FP32: vmov    r0, r1, d0              @ encoding: [0x51,0xec,0x10,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vmov d0, r0, r1
# FP32: vmov    d0, r0, r1              @ encoding: [0x41,0xec,0x10,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vmov r0, r1, s0, s1
# FP32: vmov    r0, r1, s0, s1          @ encoding: [0x51,0xec,0x10,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vmov s0, s1, r0, r1
# FP32: vmov    s0, s1, r0, r1          @ encoding: [0x41,0xec,0x10,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vmov.f32 s0, s1
# FP32: vmov.f32        s0, s1          @ encoding: [0xb0,0xee,0x60,0x0a]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers

vmov.f64 d0, d1
# FP64: vmov.f64        d0, d1          @ encoding: [0xb0,0xee,0x41,0x0b]
# NOFP64: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: 64-bit fp registers

vmov.32 r0, d1[0]
# FP32: vmov.32 r0, d1[0]               @ encoding: [0x11,0xee,0x10,0x0b]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires: fp registers

vmrs apsr_nzcv, fpscr
# FP32: vmrs    APSR_nzcv, fpscr        @ encoding: [0xf1,0xee,0x10,0xfa]
# NOFP32: :[[@LINE-2]]:{{[0-9]+}}: {{note|error}}: instruction requires: fp registers
