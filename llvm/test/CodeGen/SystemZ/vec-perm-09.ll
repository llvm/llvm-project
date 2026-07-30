; Test general vector permute of a v16i8.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-CODE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s

define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-CODE-LABEL: f1:
; CHECK-CODE: larl [[REG:%r[0-5]]],
; CHECK-CODE: vl [[MASK:%v[0-9]+]], 0([[REG]])
; CHECK-CODE: vperm %v24, %v24, %v26, [[MASK]]
; CHECK-CODE: br %r14
;
; CHECK-VECTOR: .byte 1
; CHECK-VECTOR-NEXT: .byte 19
; CHECK-VECTOR-NEXT: .byte 6
; CHECK-VECTOR-NEXT: .byte 5
; CHECK-VECTOR-NEXT: .byte 20
; CHECK-VECTOR-NEXT: .byte 22
; CHECK-VECTOR-NEXT: .byte 1
; CHECK-VECTOR-NEXT: .byte 1
; CHECK-VECTOR-NEXT: .byte 25
; CHECK-VECTOR-NEXT: .byte 29
; CHECK-VECTOR-NEXT: .byte 11
; Any byte would be OK here
; CHECK-VECTOR-NEXT: .space 1
; CHECK-VECTOR-NEXT: .byte 31
; CHECK-VECTOR-NEXT: .byte 4
; CHECK-VECTOR-NEXT: .byte 15
; CHECK-VECTOR-NEXT: .byte 19
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 1, i32 19, i32 6, i32 5,
                                   i32 20, i32 22, i32 1, i32 1,
                                   i32 25, i32 29, i32 11, i32 undef,
                                   i32 31, i32 4, i32 15, i32 19>
  ret <16 x i8> %ret
}
