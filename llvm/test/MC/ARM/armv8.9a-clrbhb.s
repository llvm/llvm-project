// CLRBHB is optional for all v8a/v9a, mandatory for 8.9a/9.4a.
// Assembly is always permitted for instructions in the hint space.

// Invalid before v8
// RUN: not llvm-mc -show-encoding -triple armv7-none-none-eabi < %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not llvm-mc -show-encoding -triple armv7-none-none-eabi -mattr=-clrbhb < %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not llvm-mc -show-encoding -triple armv7-none-none-eabi -mattr=+clrbhb < %s 2>&1 | FileCheck %s --check-prefix=REQUIRESV8

// Optional, off by default
// RUN: llvm-mc -show-encoding -triple armv8-none-none-eabi < %s | FileCheck %s --check-prefix=A32_HINT
// RUN: llvm-mc -show-encoding -triple armv8.8a-none-none-eabi < %s | FileCheck %s --check-prefix=A32_HINT
// RUN: llvm-mc -show-encoding -triple thumbv8-none-none-eabi < %s | FileCheck %s --check-prefix=T32_HINT
// RUN: llvm-mc -show-encoding -triple thumbv8.8a-none-none-eabi < %s | FileCheck %s --check-prefix=T32_HINT

// Optional, off by default, doubly disabled
// RUN: llvm-mc -show-encoding -triple armv8-none-none-eabi -mattr=-clrbhb < %s | FileCheck %s --check-prefix=A32_HINT
// RUN: llvm-mc -show-encoding -triple armv8.8a-none-none-eabi -mattr=-clrbhb < %s | FileCheck %s --check-prefix=A32_HINT
// RUN: llvm-mc -show-encoding -triple thumbv8-none-none-eabi -mattr=-clrbhb < %s | FileCheck %s --check-prefix=T32_HINT
// RUN: llvm-mc -show-encoding -triple thumbv8.8a-none-none-eabi -mattr=-clrbhb < %s | FileCheck %s --check-prefix=T32_HINT

// Optional, off by default, manually enabled
// RUN: llvm-mc -show-encoding -triple armv8-none-none-eabi -mattr=+clrbhb < %s | FileCheck %s --check-prefix=A32_CLRBHB
// RUN: llvm-mc -show-encoding -triple armv8.8a-none-none-eabi -mattr=+clrbhb < %s | FileCheck %s --check-prefix=A32_CLRBHB
// RUN: llvm-mc -show-encoding -triple thumbv8-none-none-eabi -mattr=+clrbhb < %s | FileCheck %s --check-prefix=T32_CLRBHB
// RUN: llvm-mc -show-encoding -triple thumbv8.8a-none-none-eabi -mattr=+clrbhb < %s | FileCheck %s --check-prefix=T32_CLRBHB

// Mandatory, enabled by default
// RUN: llvm-mc -show-encoding -triple armv8.9a-none-none-eabi < %s | FileCheck %s --check-prefix=A32_CLRBHB
// RUN: llvm-mc -show-encoding -triple thumbv8.9a-none-none-eabi < %s | FileCheck %s --check-prefix=T32_CLRBHB

// Mandatory, on by default, doubly enabled
// RUN: llvm-mc -show-encoding -triple armv8.9a-none-none-eabi -mattr=+clrbhb < %s | FileCheck %s --check-prefix=A32_CLRBHB
// RUN: llvm-mc -show-encoding -triple thumbv8.9a-none-none-eabi -mattr=+clrbhb < %s | FileCheck %s --check-prefix=T32_CLRBHB

// Mandatory, can't prevent disabling in LLVM
// RUN: llvm-mc -show-encoding -triple armv8.9a-none-none-eabi -mattr=-clrbhb < %s | FileCheck %s --check-prefix=A32_HINT
// RUN: llvm-mc -show-encoding -triple thumbv8.9a-none-none-eabi -mattr=-clrbhb < %s | FileCheck %s --check-prefix=T32_HINT

        clrbhb
        hint #22

// INVALID: <stdin>:[[@LINE-3]]:9: error: invalid instruction
// INVALID-NOT: <stdin>:[[@LINE-3]]:9: error: invalid instruction
// REQUIRESV8: <stdin>:[[@LINE-5]]:9: error: instruction requires: armv8
// REQUIRESV8-NOT: <stdin>:[[@LINE-5]]:9: error: instruction requires: armv8
// A32_CLRBHB: clrbhb   @ encoding: [0x16,0xf0,0x20,0xe3]
// A32_CLRBHB: clrbhb   @ encoding: [0x16,0xf0,0x20,0xe3]
// A32_HINT: hint #22   @ encoding: [0x16,0xf0,0x20,0xe3]
// A32_HINT: hint #22   @ encoding: [0x16,0xf0,0x20,0xe3]
// T32_CLRBHB: clrbhb   @ encoding: [0xaf,0xf3,0x16,0x80]
// T32_CLRBHB: clrbhb   @ encoding: [0xaf,0xf3,0x16,0x80]
// T32_HINT: hint.w #22 @ encoding: [0xaf,0xf3,0x16,0x80]
// T32_HINT: hint.w #22 @ encoding: [0xaf,0xf3,0x16,0x80]
