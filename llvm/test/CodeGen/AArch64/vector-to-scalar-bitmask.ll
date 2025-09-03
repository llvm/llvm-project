; RUN: llc -O3 -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s --check-prefix=CHECK-LE
; RUN: llc -O3 -mtriple=aarch64_be-linux-gnu %s -o - | FileCheck %s --check-prefix=CHECK-BE

@haystack4 = internal unnamed_addr constant [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 4
@haystack16 = internal unnamed_addr constant [16 x i8] [i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15], align 16


define i8 @test4() {
  %matches = alloca <4 x i1>, align 1
  %index_ptr = alloca i64, align 8
  store i64 0, ptr %index_ptr, align 8
  %index_val = load i64, ptr %index_ptr, align 8
  %haystack = getelementptr inbounds i32, ptr getelementptr inbounds (i8, ptr @haystack4, i64 0), i64 %index_val
  %h_vec = load <4 x i32>, ptr %haystack, align 4
  %cmp_vec = icmp eq <4 x i32> %h_vec, <i32 2, i32 2, i32 2, i32 2>
  store volatile <4 x i1> %cmp_vec, ptr %matches, align 1
  %cmp_load = load volatile <4 x i1>, ptr %matches, align 1
  %extr = extractelement <4 x i1> %cmp_load, i64 2
  %ret = zext i1 %extr to i8
  ret i8 %ret
}

define i8 @test16() {
  %matches = alloca <16 x i1>, align 2
  %index_ptr = alloca i64, align 8
  store i64 0, ptr %index_ptr, align 8
  %index_val = load i64, ptr %index_ptr, align 8
  %haystack = getelementptr inbounds i8, ptr getelementptr inbounds (i8, ptr @haystack16, i64 0), i64 %index_val
  %h_vec = load <16 x i8>, ptr %haystack, align 16
  %cmp_vec = icmp eq <16 x i8> %h_vec, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  store volatile <16 x i1> %cmp_vec, ptr %matches, align 2
  %cmp_load = load volatile <16 x i1>, ptr %matches, align 2
  %extr = extractelement <16 x i1> %cmp_load, i64 7
  %ret = zext i1 %extr to i8
  ret i8 %ret
}

; Little endian

; CHECK-LE-LABEL: .LCPI0_0:
; CHECK-LE-NEXT: .word 1
; CHECK-LE-NEXT: .word 2
; CHECK-LE-NEXT: .word 4
; CHECK-LE-NEXT: .word 8

; CHECK-LE-LABEL: .LCPI1_0:
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


; Big endian

; CHECK-BE-LABEL: .LCPI0_0:
; CHECK-BE-NEXT: .word 8
; CHECK-BE-NEXT: .word 4
; CHECK-BE-NEXT: .word 2
; CHECK-BE-NEXT: .word 1

; CHECK-BE-LABEL: .LCPI1_0:
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
