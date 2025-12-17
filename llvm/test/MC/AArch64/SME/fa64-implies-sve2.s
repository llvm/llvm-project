// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+sme-fa64 < %s | FileCheck %s

// Verify sme-fa64 implies SVE2
ldnt1sh z0.s, p0/z, [z1.s]
// CHECK: ldnt1sh { z0.s }, p0/z, [z1.s]
