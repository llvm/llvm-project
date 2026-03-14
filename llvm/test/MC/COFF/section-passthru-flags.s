// RUN: llvm-mc -triple i386-pc-win32 < %s | FileCheck %s
.section .klaatu,"wn"
// CHECK: .section .klaatu,"wn"
.section .barada,"y"
// CHECK: .section .barada,"y"
.section .nikto,"dws"
// CHECK: .section .nikto,"dws"
