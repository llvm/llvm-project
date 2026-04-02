; RUN: llc -O3 -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s --check-prefix=CHECK-LE
; RUN: llc -O3 -mtriple=aarch64_be-linux-gnu %s -o - | FileCheck %s --check-prefix=CHECK-BE

define i16 @convert_to_bitmask16(<16 x i8> %vec) {
  %cmp_result = icmp ne <16 x i8> %vec, zeroinitializer
  %bitmask = bitcast <16 x i1> %cmp_result to i16
  ret i16 %bitmask
}

define i16 @convert_to_bitmask8(<8 x i16> %vec) {
  %cmp_result = icmp ne <8 x i16> %vec, zeroinitializer
  %bitmask = bitcast <8 x i1> %cmp_result to i8
  %extended_bitmask = zext i8 %bitmask to i16
  ret i16 %extended_bitmask
}

; Little endian

; CHECK-LE-LABEL: .LCPI0_0:
; CHECK-LE-NEXT: .byte 1
; CHECK-LE-NEXT: .byte 2
; CHECK-LE-NEXT: .byte 4
; CHECK-LE-NEXT: .byte 8
; CHECK-LE-NEXT: .byte 16
; CHECK-LE-NEXT: .byte 32
; CHECK-LE-NEXT: .byte 64
; CHECK-LE-NEXT: .byte 128
; CHECK-LE-NEXT: .byte 1
; CHECK-LE-NEXT: .byte 2
; CHECK-LE-NEXT: .byte 4
; CHECK-LE-NEXT: .byte 8
; CHECK-LE-NEXT: .byte 16
; CHECK-LE-NEXT: .byte 32
; CHECK-LE-NEXT: .byte 64
; CHECK-LE-NEXT: .byte 128

; CHECK-LE-LABEL: .LCPI1_0:
; CHECK-LE-NEXT: .hword 1
; CHECK-LE-NEXT: .hword 2
; CHECK-LE-NEXT: .hword 4
; CHECK-LE-NEXT: .hword 8
; CHECK-LE-NEXT: .hword 16
; CHECK-LE-NEXT: .hword 32
; CHECK-LE-NEXT: .hword 64
; CHECK-LE-NEXT: .hword 128

; Big endian

; CHECK-BE-LABEL: .LCPI0_0:
; CHECK-BE-NEXT: .byte 128
; CHECK-BE-NEXT: .byte 64
; CHECK-BE-NEXT: .byte 32
; CHECK-BE-NEXT: .byte 16
; CHECK-BE-NEXT: .byte 8
; CHECK-BE-NEXT: .byte 4
; CHECK-BE-NEXT: .byte 2
; CHECK-BE-NEXT: .byte 1
; CHECK-BE-NEXT: .byte 128
; CHECK-BE-NEXT: .byte 64
; CHECK-BE-NEXT: .byte 32
; CHECK-BE-NEXT: .byte 16
; CHECK-BE-NEXT: .byte 8
; CHECK-BE-NEXT: .byte 4
; CHECK-BE-NEXT: .byte 2
; CHECK-BE-NEXT: .byte 1

; CHECK-BE-LABEL: .LCPI1_0:
; CHECK-BE-NEXT: .hword 128
; CHECK-BE-NEXT: .hword 64
; CHECK-BE-NEXT: .hword 32
; CHECK-BE-NEXT: .hword 16
; CHECK-BE-NEXT: .hword 8
; CHECK-BE-NEXT: .hword 4
; CHECK-BE-NEXT: .hword 2
; CHECK-BE-NEXT: .hword 1
