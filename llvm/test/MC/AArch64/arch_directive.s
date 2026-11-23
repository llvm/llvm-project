# RUN: llvm-mc -assemble -triple=aarch64- %s | FileCheck %s
# CHECK: .text
# CHECK-NEXT: .arch armv8-a+lse
# CHECK-NEXT: cas x0, x1, [x2]
# CHECK-NEXT: .arch armv8-a
# CHECK-NEXT: .arch_extension lse
# CHECK-NEXT: cas x0, x1, [x2]
# CHECK-NEXT: .arch_extension nolse
.text
.arch armv8-a+lse
cas x0, x1, [x2]
.arch armv8-a
.arch_extension lse
cas x0, x1, [x2]
.arch_extension nolse
