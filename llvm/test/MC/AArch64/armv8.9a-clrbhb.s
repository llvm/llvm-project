// CLRBHB is optional for all v8a/v9a, mandatory for 8.9a/9.4a.
// Assembly is always permitted for instructions in the hint space.

// Optional, off by default
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8a < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8.8a < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9a < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9.3a < %s | FileCheck %s --check-prefix=HINT_22

// Optional, off by default, doubly disabled
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8.8a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9.3a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22

// Optional, off by default, manually enabled
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+clrbhb < %s | FileCheck %s --check-prefix=CLRBHB
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8a,+clrbhb < %s | FileCheck %s --check-prefix=CLRBHB
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8.8a,+clrbhb < %s | FileCheck %s --check-prefix=CLRBHB
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9a,+clrbhb < %s | FileCheck %s --check-prefix=CLRBHB
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9.3a,+clrbhb < %s | FileCheck %s --check-prefix=CLRBHB

// Mandatory, enabled by default
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8.9a < %s | FileCheck %s --check-prefix=CLRBHB
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9.4a < %s | FileCheck %s --check-prefix=CLRBHB

// Mandatory, on by default, doubly enabled
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8.9a,+clrbhb < %s | FileCheck %s --check-prefix=CLRBHB
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9.4a,+clrbhb < %s | FileCheck %s --check-prefix=CLRBHB

// Mandatory, can't prevent disabling in LLVM
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v8.9a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64-none-none-eabi -mattr=+v9.4a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22

        clrbhb
        hint #22

// CLRBHB: clrbhb    // encoding: [0xdf,0x22,0x03,0xd5]
// CLRBHB: clrbhb    // encoding: [0xdf,0x22,0x03,0xd5]
// HINT_22: hint #22 // encoding: [0xdf,0x22,0x03,0xd5]
// HINT_22: hint #22 // encoding: [0xdf,0x22,0x03,0xd5]
