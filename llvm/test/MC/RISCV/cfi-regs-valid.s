# RUN: llvm-mc %s -triple=riscv32 | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 | FileCheck %s

.cfi_startproc
# CHECK: .cfi_offset zero, 0
.cfi_offset x0, 0
# CHECK: .cfi_offset ra, 8
.cfi_offset x1, 8
# CHECK: .cfi_offset sp, 16
.cfi_offset x2, 16
# CHECK: .cfi_offset gp, 24
.cfi_offset x3, 24
# CHECK: .cfi_offset tp, 32
.cfi_offset x4, 32
# CHECK: .cfi_offset t0, 40
.cfi_offset x5, 40
# CHECK: .cfi_offset t1, 48
.cfi_offset x6, 48
# CHECK: .cfi_offset t2, 56
.cfi_offset x7, 56
# CHECK: .cfi_offset s0, 64
.cfi_offset x8, 64
# CHECK: .cfi_offset s1, 72
.cfi_offset x9, 72
# CHECK: .cfi_offset a0, 80
.cfi_offset x10, 80
# CHECK: .cfi_offset a1, 88
.cfi_offset x11, 88
# CHECK: .cfi_offset a2, 96
.cfi_offset x12, 96
# CHECK: .cfi_offset a3, 104
.cfi_offset x13, 104
# CHECK: .cfi_offset a4, 112
.cfi_offset x14, 112
# CHECK: .cfi_offset a5, 120
.cfi_offset x15, 120
# CHECK: .cfi_offset a6, 128
.cfi_offset x16, 128
# CHECK: .cfi_offset a7, 136
.cfi_offset x17, 136
# CHECK: .cfi_offset s2, 144
.cfi_offset x18, 144
# CHECK: .cfi_offset s3, 152
.cfi_offset x19, 152
# CHECK: .cfi_offset s4, 160
.cfi_offset x20, 160
# CHECK: .cfi_offset s5, 168
.cfi_offset x21, 168
# CHECK: .cfi_offset s6, 176
.cfi_offset x22, 176
# CHECK: .cfi_offset s7, 184
.cfi_offset x23, 184
# CHECK: .cfi_offset s8, 192
.cfi_offset x24, 192
# CHECK: .cfi_offset s9, 200
.cfi_offset x25, 200
# CHECK: .cfi_offset s10, 208
.cfi_offset x26, 208
# CHECK: .cfi_offset s11, 216
.cfi_offset x27, 216
# CHECK: .cfi_offset t3, 224
.cfi_offset x28, 224
# CHECK: .cfi_offset t4, 232
.cfi_offset x29, 232
# CHECK: .cfi_offset t5, 240
.cfi_offset x30, 240
# CHECK: .cfi_offset t6, 248
.cfi_offset x31, 248
.cfi_endproc

.cfi_startproc
# CHECK: .cfi_offset zero, 0
.cfi_offset zero, 0
# CHECK: .cfi_offset ra, 8
.cfi_offset ra, 8
# CHECK: .cfi_offset sp, 16
.cfi_offset sp, 16
# CHECK: .cfi_offset gp, 24
.cfi_offset gp, 24
# CHECK: .cfi_offset tp, 32
.cfi_offset tp, 32
# CHECK: .cfi_offset t0, 40
.cfi_offset t0, 40
# CHECK: .cfi_offset t1, 48
.cfi_offset t1, 48
# CHECK: .cfi_offset t2, 56
.cfi_offset t2, 56
# CHECK: .cfi_offset s0, 64
.cfi_offset s0, 64
# CHECK: .cfi_offset s1, 72
.cfi_offset s1, 72
# CHECK: .cfi_offset a0, 80
.cfi_offset a0, 80
# CHECK: .cfi_offset a1, 88
.cfi_offset a1, 88
# CHECK: .cfi_offset a2, 96
.cfi_offset a2, 96
# CHECK: .cfi_offset a3, 104
.cfi_offset a3, 104
# CHECK: .cfi_offset a4, 112
.cfi_offset a4, 112
# CHECK: .cfi_offset a5, 120
.cfi_offset a5, 120
# CHECK: .cfi_offset a6, 128
.cfi_offset a6, 128
# CHECK: .cfi_offset a7, 136
.cfi_offset a7, 136
# CHECK: .cfi_offset s2, 144
.cfi_offset s2, 144
# CHECK: .cfi_offset s3, 152
.cfi_offset s3, 152
# CHECK: .cfi_offset s4, 160
.cfi_offset s4, 160
# CHECK: .cfi_offset s5, 168
.cfi_offset s5, 168
# CHECK: .cfi_offset s6, 176
.cfi_offset s6, 176
# CHECK: .cfi_offset s7, 184
.cfi_offset s7, 184
# CHECK: .cfi_offset s8, 192
.cfi_offset s8, 192
# CHECK: .cfi_offset s9, 200
.cfi_offset s9, 200
# CHECK: .cfi_offset s10, 208
.cfi_offset s10, 208
# CHECK: .cfi_offset s11, 216
.cfi_offset s11, 216
# CHECK: .cfi_offset t3, 224
.cfi_offset t3, 224
# CHECK: .cfi_offset t4, 232
.cfi_offset t4, 232
# CHECK: .cfi_offset t5, 240
.cfi_offset t5, 240
# CHECK: .cfi_offset t6, 248
.cfi_offset t6, 248

.cfi_endproc
