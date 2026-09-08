// RUN: llvm-mc -triple aarch64_lfi %s 2>&1 | FileCheck %s --implicit-check-not=warning

clrex
// CHECK: clrex
clrex #7
// CHECK: clrex #7
hint #7
// CHECK: hint #7
hint #127
// CHECK: hint #127
nop
// CHECK: nop
dmb sy
// CHECK: dmb sy
dmb ish
// CHECK: dmb ish
dsb sy
// CHECK: dsb sy
isb
// CHECK: isb
yield
// CHECK: yield
wfe
// CHECK: wfe
wfi
// CHECK: wfi
sev
// CHECK: sev
sevl
// CHECK: sevl
