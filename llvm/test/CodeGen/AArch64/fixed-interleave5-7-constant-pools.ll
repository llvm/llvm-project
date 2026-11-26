; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s --match-full-lines

; CHECK-LABEL: .LCPI0_0:
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

; CHECK-LABEL: .LCPI0_1:
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

; CHECK-LABEL: .LCPI0_2:
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

; CHECK-LABEL: .LCPI0_3:
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

; CHECK-LABEL: .LCPI0_4:
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

; CHECK-LABEL: .LCPI0_5:
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

; CHECK-LABEL: .LCPI0_6:
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

; CHECK-LABEL: .LCPI0_7:
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

; CHECK-LABEL: .LCPI0_8:
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

; CHECK-LABEL: .LCPI0_9:
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

; CHECK-LABEL: .LCPI1_0:
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

; CHECK-LABEL: .LCPI1_1:
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

; CHECK-LABEL: .LCPI1_2:
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

; CHECK-LABEL: .LCPI1_3:
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

; CHECK-LABEL: .LCPI1_4:
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

; CHECK-LABEL: .LCPI1_5:
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

; CHECK-LABEL: .LCPI1_6:
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

; CHECK-LABEL: .LCPI1_7:
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

; CHECK-LABEL: .LCPI1_8:
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

; CHECK-LABEL: .LCPI1_9:
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

; CHECK-LABEL: .LCPI2_0:
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 48 // 0x30
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 49 // 0x31
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 18 // 0x12

; CHECK-LABEL: .LCPI2_1:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI2_2:
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 50 // 0x32
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 35 // 0x23
; CHECK-NEXT: .byte 51 // 0x33
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 52 // 0x34

; CHECK-LABEL: .LCPI2_3:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 18 // 0x12
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 35 // 0x23
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI2_4:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 37 // 0x25
; CHECK-NEXT: .byte 53 // 0x35
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 22 // 0x16
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 54 // 0x36
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI2_5:
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 37 // 0x25
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 22 // 0x16

; CHECK-LABEL: .LCPI2_6:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 39 // 0x27
; CHECK-NEXT: .byte 55 // 0x37
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 56 // 0x38
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 9 // 0x9

; CHECK-LABEL: .LCPI2_7:
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 39 // 0x27
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI2_8:
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 41 // 0x29
; CHECK-NEXT: .byte 57 // 0x39
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 42 // 0x2a
; CHECK-NEXT: .byte 58 // 0x3a
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 43 // 0x2b

; CHECK-LABEL: .LCPI2_9:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 41 // 0x29
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 42 // 0x2a
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI2_10:
; CHECK-NEXT: .byte 59 // 0x3b
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 44 // 0x2c
; CHECK-NEXT: .byte 60 // 0x3c
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 29 // 0x1d
; CHECK-NEXT: .byte 45 // 0x2d
; CHECK-NEXT: .byte 61 // 0x3d
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI2_11:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 43 // 0x2b
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 44 // 0x2c
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 13 // 0xd

; CHECK-LABEL: .LCPI2_12:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 46 // 0x2e
; CHECK-NEXT: .byte 62 // 0x3e
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 47 // 0x2f
; CHECK-NEXT: .byte 63 // 0x3f
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI2_13:
; CHECK-NEXT: .byte 29 // 0x1d
; CHECK-NEXT: .byte 45 // 0x2d
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 46 // 0x2e
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 47 // 0x2f

define void @interleave7_i8(ptr %src, ptr %dst) {
  %p1 = getelementptr i8, ptr %src, i64 16
  %p2 = getelementptr i8, ptr %src, i64 32
  %p3 = getelementptr i8, ptr %src, i64 48
  %p4 = getelementptr i8, ptr %src, i64 64
  %p5 = getelementptr i8, ptr %src, i64 80
  %p6 = getelementptr i8, ptr %src, i64 96
  %v0 = load <16 x i8>, ptr %src, align 16
  %v1 = load <16 x i8>, ptr %p1, align 16
  %v2 = load <16 x i8>, ptr %p2, align 16
  %v3 = load <16 x i8>, ptr %p3, align 16
  %v4 = load <16 x i8>, ptr %p4, align 16
  %v5 = load <16 x i8>, ptr %p5, align 16
  %v6 = load <16 x i8>, ptr %p6, align 16
  %wide = call <112 x i8> @llvm.vector.interleave7.v112i8(<16 x i8> %v0, <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3, <16 x i8> %v4, <16 x i8> %v5, <16 x i8> %v6)
  store <112 x i8> %wide, ptr %dst, align 16
  ret void
}

; CHECK-LABEL: .LCPI3_0:
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 41 // 0x29
; CHECK-NEXT: .byte 48 // 0x30
; CHECK-NEXT: .byte 55 // 0x37
; CHECK-NEXT: .byte 62 // 0x3e
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI3_1:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 47 // 0x2f

; CHECK-LABEL: .LCPI3_2:
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 47 // 0x2f
; CHECK-NEXT: .byte 54 // 0x36
; CHECK-NEXT: .byte 61 // 0x3d
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI3_3:
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
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 18 // 0x12
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 39 // 0x27
; CHECK-NEXT: .byte 46 // 0x2e

; CHECK-LABEL: .LCPI3_4:
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 18 // 0x12
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 39 // 0x27
; CHECK-NEXT: .byte 46 // 0x2e
; CHECK-NEXT: .byte 53 // 0x35
; CHECK-NEXT: .byte 60 // 0x3c
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI3_5:
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
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 45 // 0x2d

; CHECK-LABEL: .LCPI3_6:
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 45 // 0x2d
; CHECK-NEXT: .byte 52 // 0x34
; CHECK-NEXT: .byte 59 // 0x3b
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI3_7:
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
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 37 // 0x25
; CHECK-NEXT: .byte 44 // 0x2c

; CHECK-LABEL: .LCPI3_8:
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 37 // 0x25
; CHECK-NEXT: .byte 44 // 0x2c
; CHECK-NEXT: .byte 51 // 0x33
; CHECK-NEXT: .byte 58 // 0x3a
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI3_9:
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
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 22 // 0x16
; CHECK-NEXT: .byte 29 // 0x1d
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 43 // 0x2b

; CHECK-LABEL: .LCPI3_10:
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 22 // 0x16
; CHECK-NEXT: .byte 29 // 0x1d
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 43 // 0x2b
; CHECK-NEXT: .byte 50 // 0x32
; CHECK-NEXT: .byte 57 // 0x39
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI3_11:
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
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 35 // 0x23
; CHECK-NEXT: .byte 42 // 0x2a

; CHECK-LABEL: .LCPI3_12:
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 35 // 0x23
; CHECK-NEXT: .byte 42 // 0x2a
; CHECK-NEXT: .byte 49 // 0x31
; CHECK-NEXT: .byte 56 // 0x38
; CHECK-NEXT: .byte 63 // 0x3f
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI3_13:
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
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 41 // 0x29

define void @deinterleave7_i8(ptr %src, ptr %dst) {
  %wide = load <112 x i8>, ptr %src, align 16
  %parts = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.vector.deinterleave7.v112i8(<112 x i8> %wide)
  %v0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 0
  %v1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 1
  %v2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 2
  %v3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 3
  %v4 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 4
  %v5 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 5
  %v6 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %parts, 6
  %p1 = getelementptr i8, ptr %dst, i64 16
  %p2 = getelementptr i8, ptr %dst, i64 32
  %p3 = getelementptr i8, ptr %dst, i64 48
  %p4 = getelementptr i8, ptr %dst, i64 64
  %p5 = getelementptr i8, ptr %dst, i64 80
  %p6 = getelementptr i8, ptr %dst, i64 96
  store <16 x i8> %v0, ptr %dst, align 16
  store <16 x i8> %v1, ptr %p1, align 16
  store <16 x i8> %v2, ptr %p2, align 16
  store <16 x i8> %v3, ptr %p3, align 16
  store <16 x i8> %v4, ptr %p4, align 16
  store <16 x i8> %v5, ptr %p5, align 16
  store <16 x i8> %v6, ptr %p6, align 16
  ret void
}

; CHECK-LABEL: .LCPI4_0:
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 48 // 0x30
; CHECK-NEXT: .byte 49 // 0x31
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 18 // 0x12
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 35 // 0x23

; CHECK-LABEL: .LCPI4_1:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI4_2:
; CHECK-NEXT: .byte 50 // 0x32
; CHECK-NEXT: .byte 51 // 0x33
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 37 // 0x25
; CHECK-NEXT: .byte 52 // 0x34
; CHECK-NEXT: .byte 53 // 0x35
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 7 // 0x7

; CHECK-LABEL: .LCPI4_3:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI4_4:
; CHECK-NEXT: .byte 22 // 0x16
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 39 // 0x27
; CHECK-NEXT: .byte 54 // 0x36
; CHECK-NEXT: .byte 55 // 0x37
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 41 // 0x29
; CHECK-NEXT: .byte 56 // 0x38
; CHECK-NEXT: .byte 57 // 0x39

; CHECK-LABEL: .LCPI4_5:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI4_6:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 42 // 0x2a
; CHECK-NEXT: .byte 43 // 0x2b
; CHECK-NEXT: .byte 58 // 0x3a
; CHECK-NEXT: .byte 59 // 0x3b
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 29 // 0x1d

; CHECK-LABEL: .LCPI4_7:
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI4_8:
; CHECK-NEXT: .byte 44 // 0x2c
; CHECK-NEXT: .byte 45 // 0x2d
; CHECK-NEXT: .byte 60 // 0x3c
; CHECK-NEXT: .byte 61 // 0x3d
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 46 // 0x2e
; CHECK-NEXT: .byte 47 // 0x2f
; CHECK-NEXT: .byte 62 // 0x3e
; CHECK-NEXT: .byte 63 // 0x3f
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI4_9:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 15 // 0xf

define void @interleave5_v8i16(ptr %src, ptr %dst) {
  %p1 = getelementptr i8, ptr %src, i64 16
  %p2 = getelementptr i8, ptr %src, i64 32
  %p3 = getelementptr i8, ptr %src, i64 48
  %p4 = getelementptr i8, ptr %src, i64 64
  %v0 = load <8 x i16>, ptr %src, align 16
  %v1 = load <8 x i16>, ptr %p1, align 16
  %v2 = load <8 x i16>, ptr %p2, align 16
  %v3 = load <8 x i16>, ptr %p3, align 16
  %v4 = load <8 x i16>, ptr %p4, align 16
  %wide = call <40 x i16> @llvm.vector.interleave5.v40i16(<8 x i16> %v0, <8 x i16> %v1, <8 x i16> %v2, <8 x i16> %v3, <8 x i16> %v4)
  store <40 x i16> %wide, ptr %dst, align 16
  ret void
}

; CHECK-LABEL: .LCPI5_0:
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 41 // 0x29
; CHECK-NEXT: .byte 54 // 0x36
; CHECK-NEXT: .byte 55 // 0x37
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI5_1:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 18 // 0x12
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 46 // 0x2e
; CHECK-NEXT: .byte 47 // 0x2f

; CHECK-LABEL: .LCPI5_2:
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 39 // 0x27
; CHECK-NEXT: .byte 52 // 0x34
; CHECK-NEXT: .byte 53 // 0x35
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI5_3:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 44 // 0x2c
; CHECK-NEXT: .byte 45 // 0x2d

; CHECK-LABEL: .LCPI5_4:
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 22 // 0x16
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 37 // 0x25
; CHECK-NEXT: .byte 50 // 0x32
; CHECK-NEXT: .byte 51 // 0x33
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI5_5:
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 29 // 0x1d
; CHECK-NEXT: .byte 42 // 0x2a
; CHECK-NEXT: .byte 43 // 0x2b

; CHECK-LABEL: .LCPI5_6:
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 35 // 0x23
; CHECK-NEXT: .byte 48 // 0x30
; CHECK-NEXT: .byte 49 // 0x31
; CHECK-NEXT: .byte 62 // 0x3e
; CHECK-NEXT: .byte 63 // 0x3f
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI5_7:
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
; CHECK-NEXT: .byte 12 // 0xc
; CHECK-NEXT: .byte 13 // 0xd
; CHECK-NEXT: .byte 26 // 0x1a
; CHECK-NEXT: .byte 27 // 0x1b
; CHECK-NEXT: .byte 40 // 0x28
; CHECK-NEXT: .byte 41 // 0x29

; CHECK-LABEL: .LCPI5_8:
; CHECK-NEXT: .byte 4 // 0x4
; CHECK-NEXT: .byte 5 // 0x5
; CHECK-NEXT: .byte 18 // 0x12
; CHECK-NEXT: .byte 19 // 0x13
; CHECK-NEXT: .byte 32 // 0x20
; CHECK-NEXT: .byte 33 // 0x21
; CHECK-NEXT: .byte 46 // 0x2e
; CHECK-NEXT: .byte 47 // 0x2f
; CHECK-NEXT: .byte 60 // 0x3c
; CHECK-NEXT: .byte 61 // 0x3d
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI5_9:
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
; CHECK-NEXT: .byte 10 // 0xa
; CHECK-NEXT: .byte 11 // 0xb
; CHECK-NEXT: .byte 24 // 0x18
; CHECK-NEXT: .byte 25 // 0x19
; CHECK-NEXT: .byte 38 // 0x26
; CHECK-NEXT: .byte 39 // 0x27

; CHECK-LABEL: .LCPI5_10:
; CHECK-NEXT: .byte 2 // 0x2
; CHECK-NEXT: .byte 3 // 0x3
; CHECK-NEXT: .byte 16 // 0x10
; CHECK-NEXT: .byte 17 // 0x11
; CHECK-NEXT: .byte 30 // 0x1e
; CHECK-NEXT: .byte 31 // 0x1f
; CHECK-NEXT: .byte 44 // 0x2c
; CHECK-NEXT: .byte 45 // 0x2d
; CHECK-NEXT: .byte 58 // 0x3a
; CHECK-NEXT: .byte 59 // 0x3b
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI5_11:
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
; CHECK-NEXT: .byte 8 // 0x8
; CHECK-NEXT: .byte 9 // 0x9
; CHECK-NEXT: .byte 22 // 0x16
; CHECK-NEXT: .byte 23 // 0x17
; CHECK-NEXT: .byte 36 // 0x24
; CHECK-NEXT: .byte 37 // 0x25

; CHECK-LABEL: .LCPI5_12:
; CHECK-NEXT: .byte 0 // 0x0
; CHECK-NEXT: .byte 1 // 0x1
; CHECK-NEXT: .byte 14 // 0xe
; CHECK-NEXT: .byte 15 // 0xf
; CHECK-NEXT: .byte 28 // 0x1c
; CHECK-NEXT: .byte 29 // 0x1d
; CHECK-NEXT: .byte 42 // 0x2a
; CHECK-NEXT: .byte 43 // 0x2b
; CHECK-NEXT: .byte 56 // 0x38
; CHECK-NEXT: .byte 57 // 0x39
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff
; CHECK-NEXT: .byte 255 // 0xff

; CHECK-LABEL: .LCPI5_13:
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
; CHECK-NEXT: .byte 6 // 0x6
; CHECK-NEXT: .byte 7 // 0x7
; CHECK-NEXT: .byte 20 // 0x14
; CHECK-NEXT: .byte 21 // 0x15
; CHECK-NEXT: .byte 34 // 0x22
; CHECK-NEXT: .byte 35 // 0x23

define void @deinterleave7_v8i16(ptr %src, ptr %dst) {
  %wide = load <56 x i16>, ptr %src, align 16
  %parts = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.vector.deinterleave7.v56i16(<56 x i16> %wide)
  %v0 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 0
  %v1 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 1
  %v2 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 2
  %v3 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 3
  %v4 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 4
  %v5 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 5
  %v6 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %parts, 6
  %p1 = getelementptr i8, ptr %dst, i64 16
  %p2 = getelementptr i8, ptr %dst, i64 32
  %p3 = getelementptr i8, ptr %dst, i64 48
  %p4 = getelementptr i8, ptr %dst, i64 64
  %p5 = getelementptr i8, ptr %dst, i64 80
  %p6 = getelementptr i8, ptr %dst, i64 96
  store <8 x i16> %v0, ptr %dst, align 16
  store <8 x i16> %v1, ptr %p1, align 16
  store <8 x i16> %v2, ptr %p2, align 16
  store <8 x i16> %v3, ptr %p3, align 16
  store <8 x i16> %v4, ptr %p4, align 16
  store <8 x i16> %v5, ptr %p5, align 16
  store <8 x i16> %v6, ptr %p6, align 16
  ret void
}

