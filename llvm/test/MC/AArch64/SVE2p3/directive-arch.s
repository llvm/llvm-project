// RUN: llvm-mc -triple=aarch64 < %s | FileCheck %s

.arch armv9-a+sve2p3+fpv2
fcmtl p15.s, z31.s, z31.s
// CHECK: fcmtl p15.s, z31.s, z31.s
