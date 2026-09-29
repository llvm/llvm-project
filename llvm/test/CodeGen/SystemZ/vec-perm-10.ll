; Test general vector permute of a v8i16.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-CODE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s

define <8 x i16> @f1(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-CODE-LABEL: f1:
; CHECK-CODE: larl [[REG:%r[0-5]]],
; CHECK-CODE: vl [[MASK:%v[0-9]+]], 0([[REG]])
; CHECK-CODE: vperm %v24, %v26, %v24, [[MASK]]
; CHECK-CODE: br %r14
;
; CHECK-VECTOR: .byte 0
; CHECK-VECTOR-NEXT: .byte 1
; CHECK-VECTOR-NEXT: .byte 26
; CHECK-VECTOR-NEXT: .byte 27
; Any 2 bytes would be OK here
; CHECK-VECTOR-NEXT: .space 1
; CHECK-VECTOR-NEXT: .space 1
; CHECK-VECTOR-NEXT: .byte 28
; CHECK-VECTOR-NEXT: .byte 29
; CHECK-VECTOR-NEXT: .byte 6
; CHECK-VECTOR-NEXT: .byte 7
; CHECK-VECTOR-NEXT: .byte 14
; CHECK-VECTOR-NEXT: .byte 15
; CHECK-VECTOR-NEXT: .byte 8
; CHECK-VECTOR-NEXT: .byte 9
; CHECK-VECTOR-NEXT: .byte 16
; CHECK-VECTOR-NEXT: .byte 17
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 8, i32 5, i32 undef, i32 6,
                                  i32 11, i32 15, i32 12, i32 0>
  ret <8 x i16> %ret
}
