// RUN: llvm-mc -triple=aarch64 < %s | FileCheck %s

.arch_extension sve2p3
.arch_extension fpv2
fcmtl p15.s, z31.s, z31.s
// CHECK: fcmtl p15.s, z31.s, z31.s
