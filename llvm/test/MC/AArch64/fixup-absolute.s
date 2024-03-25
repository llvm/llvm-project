// RUN: llvm-mc -triple aarch64 -filetype obj < %s -o - | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

onepart_before = 0x1234
twopart_before = 0x12345678
threepart_before = 0x1234567890AB
fourpart_before = 0x1234567890ABCDEF

// CHECK: mov     x0, #1311673391471656960
// CHECK: mov     x0, #1311673391471656960
movz x0, #:abs_g3:fourpart_before
movz x0, #:abs_g3:fourpart_after
// CHECK: mov     x0, #20014547599360
// CHECK: mov     x0, #20014547599360
movz x0, #:abs_g2:threepart_before
movz x0, #:abs_g2:threepart_after
// CHECK: movk    x0, #22136, lsl #32
// CHECK: movk    x0, #22136, lsl #32
movk x0, #:abs_g2_nc:fourpart_before
movk x0, #:abs_g2_nc:fourpart_after
// CHECK: mov     x0, #305397760
// CHECK: mov     x0, #305397760
movz x0, #:abs_g1:twopart_before
movz x0, #:abs_g1:twopart_after
// CHECK: movk    x0, #37035, lsl #16
// CHECK: movk    x0, #37035, lsl #16
movk x0, #:abs_g1_nc:fourpart_before
movk x0, #:abs_g1_nc:fourpart_after
// CHECK: mov     x0, #4660
// CHECK: mov     x0, #4660
movz x0, #:abs_g0:onepart_before
movz x0, #:abs_g0:onepart_after
// CHECK: movk    x0, #52719
// CHECK: movk    x0, #52719
movk x0, #:abs_g0_nc:fourpart_before
movk x0, #:abs_g0_nc:fourpart_after

onepart_after = 0x1234
twopart_after = 0x12345678
threepart_after = 0x1234567890AB
fourpart_after = 0x1234567890ABCDEF
