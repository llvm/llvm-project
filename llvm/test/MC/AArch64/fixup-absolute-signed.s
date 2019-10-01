// RUN: llvm-mc -triple aarch64--none-eabi -filetype obj < %s -o - | llvm-objdump -d - | FileCheck %s

onepart_before = 12345
twopart_before = -12345678
threepart_before = -1234567890

// CHECK: movn     x0, #0, lsl #32
// CHECK: movn     x0, #0, lsl #32
movz x0, #:abs_g2_s:threepart_before
movz x0, #:abs_g2_s:threepart_after

// CHECK: movk    x0, #65535, lsl #32
// CHECK: movk    x0, #65535, lsl #32
movk x0, #:abs_g2_nc:threepart_before
movk x0, #:abs_g2_nc:threepart_after

// CHECK: mov     x0, #-12320769
// CHECK: mov     x0, #-12320769
movz x0, #:abs_g1_s:twopart_before
movz x0, #:abs_g1_s:twopart_after

// CHECK: movk    x0, #46697, lsl #16
// CHECK: movk    x0, #46697, lsl #16
movk x0, #:abs_g1_nc:threepart_before
movk x0, #:abs_g1_nc:threepart_after

// CHECK: mov     x0, #12345
// CHECK: mov     x0, #12345
movz x0, #:abs_g0_s:onepart_before
movz x0, #:abs_g0_s:onepart_after

// CHECK: movk    x0, #64814
// CHECK: movk    x0, #64814
movk x0, #:abs_g0_nc:threepart_before
movk x0, #:abs_g0_nc:threepart_after

// CHECK: mov     x0, #12345
// CHECK: mov     x0, #12345
movn x0, #:abs_g0_s:onepart_before
movn x0, #:abs_g0_s:onepart_after

onepart_after = 12345
twopart_after = -12345678
threepart_after = -1234567890
