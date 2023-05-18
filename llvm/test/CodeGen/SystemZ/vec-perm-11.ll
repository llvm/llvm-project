; Test general vector permute of a v4i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-CODE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s

define <4 x i32> @f1(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-CODE-LABEL: f1:
; CHECK-CODE: larl [[REG:%r[0-5]]],
; CHECK-CODE: vl [[MASK:%v[0-9]+]], 0([[REG]])
; CHECK-CODE: vperm %v24, %v26, %v24, [[MASK]]
; CHECK-CODE: br %r14
;
; CHECK-VECTOR: .byte 4
; CHECK-VECTOR-NEXT: .byte 5
; CHECK-VECTOR-NEXT: .byte 6
; CHECK-VECTOR-NEXT: .byte 7
; CHECK-VECTOR-NEXT: .byte 20
; CHECK-VECTOR-NEXT: .byte 21
; CHECK-VECTOR-NEXT: .byte 22
; CHECK-VECTOR-NEXT: .byte 23
; Any 4 bytes would be OK here
; CHECK-VECTOR-NEXT: .space 1
; CHECK-VECTOR-NEXT: .space 1
; CHECK-VECTOR-NEXT: .space 1
; CHECK-VECTOR-NEXT: .space 1
; CHECK-VECTOR-NEXT: .byte 12
; CHECK-VECTOR-NEXT: .byte 13
; CHECK-VECTOR-NEXT: .byte 14
; CHECK-VECTOR-NEXT: .byte 15
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 5, i32 1, i32 undef, i32 7>
  ret <4 x i32> %ret
}
