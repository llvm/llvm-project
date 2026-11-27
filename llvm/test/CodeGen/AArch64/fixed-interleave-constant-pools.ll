; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s

; CHECK: [[$CPI0_0:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 48 // 0x30
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 49 // 0x31
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 18 // 0x12
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 50 // 0x32
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 3 // 0x3

; CHECK: [[$CPI0_1:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI0_2:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 35 // 0x23
; CHECK-NEXT: .byte 51 // 0x33
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 52 // 0x34
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 37 // 0x25
; CHECK-NEXT: .byte 53 // 0x35
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 22 // 0x16

; CHECK: [[$CPI0_3:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI0_4:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 54 // 0x36
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 39 // 0x27
; CHECK-NEXT: .byte 55 // 0x37
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 56 // 0x38
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 41 // 0x29

; CHECK: [[$CPI0_5:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI0_6:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 57 // 0x39
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 42 // 0x2a
; CHECK-NEXT: .byte 58 // 0x3a
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 43 // 0x2b
; CHECK-NEXT: .byte 59 // 0x3b
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 44 // 0x2c
; CHECK-NEXT: .byte 60 // 0x3c

; CHECK: [[$CPI0_7:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI0_8:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 29 // 0x1d
; CHECK-NEXT: .byte 45 // 0x2d
; CHECK-NEXT: .byte 61 // 0x3d
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 46 // 0x2e
; CHECK-NEXT: .byte 62 // 0x3e
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 47 // 0x2f
; CHECK-NEXT: .byte 63 // 0x3f
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI0_9:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 15 // 0xf

define void @interleave5_i8(ptr %src, ptr %dst) {
; CHECK-LABEL: interleave5_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp x8, [[$CPI0_0]]
; CHECK-NEXT:    ldp q0, q1, [x0]
; CHECK-NEXT:    ldr q4, [x8, :lo12:[[$CPI0_0]]]
; CHECK-NEXT:    adrp x8, [[$CPI0_8]]
; CHECK-NEXT:    adrp x9, [[$CPI0_4]]
; CHECK-NEXT:    ldp q2, q3, [x0, #32]
; CHECK-NEXT:    ldr q5, [x8, :lo12:[[$CPI0_8]]]
; CHECK-NEXT:    adrp x8, [[$CPI0_6]]
; CHECK-NEXT:    ldr q6, [x9, :lo12:[[$CPI0_4]]]
; CHECK-NEXT:    ldr q16, [x0, #64]
; CHECK-NEXT:    ldr q7, [x8, :lo12:[[$CPI0_6]]]
; CHECK-NEXT:    adrp x8, [[$CPI0_2]]
; CHECK-NEXT:    adrp x9, [[$CPI0_9]]
; CHECK-NEXT:    ldr q17, [x8, :lo12:[[$CPI0_2]]]
; CHECK-NEXT:    adrp x8, [[$CPI0_7]]
; CHECK-NEXT:    tbl v5.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v5.16b
; CHECK-NEXT:    tbl v7.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v7.16b
; CHECK-NEXT:    tbl v4.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v4.16b
; CHECK-NEXT:    tbl v6.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v6.16b
; CHECK-NEXT:    tbl v0.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v17.16b
; CHECK-NEXT:    ldr q1, [x8, :lo12:[[$CPI0_7]]]
; CHECK-NEXT:    adrp x8, [[$CPI0_5]]
; CHECK-NEXT:    ldr q2, [x8, :lo12:[[$CPI0_5]]]
; CHECK-NEXT:    adrp x8, [[$CPI0_3]]
; CHECK-NEXT:    ldr q18, [x9, :lo12:[[$CPI0_9]]]
; CHECK-NEXT:    ldr q3, [x8, :lo12:[[$CPI0_3]]]
; CHECK-NEXT:    adrp x8, [[$CPI0_1]]
; CHECK-NEXT:    tbx v7.16b, { v16.16b }, v1.16b
; CHECK-NEXT:    ldr q1, [x8, :lo12:[[$CPI0_1]]]
; CHECK-NEXT:    tbx v5.16b, { v16.16b }, v18.16b
; CHECK-NEXT:    tbx v6.16b, { v16.16b }, v2.16b
; CHECK-NEXT:    tbx v0.16b, { v16.16b }, v3.16b
; CHECK-NEXT:    tbx v4.16b, { v16.16b }, v1.16b
; CHECK-NEXT:    stp q7, q5, [x1, #48]
; CHECK-NEXT:    stp q0, q6, [x1, #16]
; CHECK-NEXT:    str q4, [x1]
; CHECK-NEXT:    ret
  %p1 = getelementptr i8, ptr %src, i64 16
  %p2 = getelementptr i8, ptr %src, i64 32
  %p3 = getelementptr i8, ptr %src, i64 48
  %p4 = getelementptr i8, ptr %src, i64 64
  %v0 = load <16 x i8>, ptr %src, align 16
  %v1 = load <16 x i8>, ptr %p1, align 16
  %v2 = load <16 x i8>, ptr %p2, align 16
  %v3 = load <16 x i8>, ptr %p3, align 16
  %v4 = load <16 x i8>, ptr %p4, align 16
  %wide = call <80 x i8> @llvm.vector.interleave5.v80i8(<16 x i8> %v0, <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3, <16 x i8> %v4)
  store <80 x i8> %wide, ptr %dst, align 16
  ret void
}

; CHECK: [[$CPI1_0:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 29 // 0x1d
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 39 // 0x27
; CHECK-NEXT: .byte 44 // 0x2c
; CHECK-NEXT: .byte 49 // 0x31
; CHECK-NEXT: .byte 54 // 0x36
; CHECK-NEXT: .byte 59 // 0x3b
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI1_1:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 15 // 0xf

; CHECK: [[$CPI1_2:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 18 // 0x12
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 43 // 0x2b
; CHECK-NEXT: .byte 48 // 0x30
; CHECK-NEXT: .byte 53 // 0x35
; CHECK-NEXT: .byte 58 // 0x3a
; CHECK-NEXT: .byte 63 // 0x3f
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI1_3:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 14 // 0xe

; CHECK: [[$CPI1_4:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 22 // 0x16
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 37 // 0x25
; CHECK-NEXT: .byte 42 // 0x2a
; CHECK-NEXT: .byte 47 // 0x2f
; CHECK-NEXT: .byte 52 // 0x34
; CHECK-NEXT: .byte 57 // 0x39
; CHECK-NEXT: .byte 62 // 0x3e
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI1_5:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 13 // 0xd

; CHECK: [[$CPI1_6:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 41 // 0x29
; CHECK-NEXT: .byte 46 // 0x2e
; CHECK-NEXT: .byte 51 // 0x33
; CHECK-NEXT: .byte 56 // 0x38
; CHECK-NEXT: .byte 61 // 0x3d
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI1_7:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 12 // 0xc

; CHECK: [[$CPI1_8:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 35 // 0x23
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 45 // 0x2d
; CHECK-NEXT: .byte 50 // 0x32
; CHECK-NEXT: .byte 55 // 0x37
; CHECK-NEXT: .byte 60 // 0x3c
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK: [[$CPI1_9:\.LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 11 // 0xb

define void @deinterleave5_i8(ptr %src, ptr %dst) {
; CHECK-LABEL: deinterleave5_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp x8, [[$CPI1_0]]
; CHECK-NEXT:    ldp q3, q4, [x0, #48]
; CHECK-NEXT:    ldr q5, [x8, :lo12:[[$CPI1_0]]]
; CHECK-NEXT:    adrp x8, [[$CPI1_8]]
; CHECK-NEXT:    adrp x9, [[$CPI1_2]]
; CHECK-NEXT:    ldp q1, q2, [x0, #16]
; CHECK-NEXT:    ldr q6, [x8, :lo12:[[$CPI1_8]]]
; CHECK-NEXT:    adrp x8, [[$CPI1_6]]
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ldr q17, [x9, :lo12:[[$CPI1_2]]]
; CHECK-NEXT:    ldr q7, [x8, :lo12:[[$CPI1_6]]]
; CHECK-NEXT:    adrp x8, [[$CPI1_4]]
; CHECK-NEXT:    ldr q16, [x8, :lo12:[[$CPI1_4]]]
; CHECK-NEXT:    adrp x8, [[$CPI1_9]]
; CHECK-NEXT:    tbl v6.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v6.16b
; CHECK-NEXT:    ldr q18, [x8, :lo12:[[$CPI1_9]]]
; CHECK-NEXT:    tbl v7.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v7.16b
; CHECK-NEXT:    adrp x8, [[$CPI1_7]]
; CHECK-NEXT:    tbl v5.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v5.16b
; CHECK-NEXT:    tbl v16.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v16.16b
; CHECK-NEXT:    tbl v0.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v17.16b
; CHECK-NEXT:    ldr q1, [x8, :lo12:[[$CPI1_7]]]
; CHECK-NEXT:    adrp x8, [[$CPI1_5]]
; CHECK-NEXT:    ldr q2, [x8, :lo12:[[$CPI1_5]]]
; CHECK-NEXT:    adrp x8, [[$CPI1_3]]
; CHECK-NEXT:    tbx v6.16b, { v4.16b }, v18.16b
; CHECK-NEXT:    ldr q3, [x8, :lo12:[[$CPI1_3]]]
; CHECK-NEXT:    adrp x8, [[$CPI1_1]]
; CHECK-NEXT:    tbx v7.16b, { v4.16b }, v1.16b
; CHECK-NEXT:    ldr q1, [x8, :lo12:[[$CPI1_1]]]
; CHECK-NEXT:    tbx v16.16b, { v4.16b }, v2.16b
; CHECK-NEXT:    tbx v0.16b, { v4.16b }, v3.16b
; CHECK-NEXT:    tbx v5.16b, { v4.16b }, v1.16b
; CHECK-NEXT:    stp q6, q7, [x1]
; CHECK-NEXT:    stp q16, q0, [x1, #32]
; CHECK-NEXT:    str q5, [x1, #64]
; CHECK-NEXT:    ret
  %wide = load <80 x i8>, ptr %src, align 16
  %parts = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.vector.deinterleave5.v80i8(<80 x i8> %wide)
  %v0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 0
  %v1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 1
  %v2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 2
  %v3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 3
  %v4 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 4
  %p1 = getelementptr i8, ptr %dst, i64 16
  %p2 = getelementptr i8, ptr %dst, i64 32
  %p3 = getelementptr i8, ptr %dst, i64 48
  %p4 = getelementptr i8, ptr %dst, i64 64
  store <16 x i8> %v0, ptr %dst, align 16
  store <16 x i8> %v1, ptr %p1, align 16
  store <16 x i8> %v2, ptr %p2, align 16
  store <16 x i8> %v3, ptr %p3, align 16
  store <16 x i8> %v4, ptr %p4, align 16
  ret void
}

