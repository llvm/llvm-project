; Test inserting a truncated value into a vector element
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-CODE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s

define <4 x i32> @f1(<4 x i32> %x, i64 %y) {
; CHECK-CODE-LABEL: f1:
; CHECK-CODE-DAG: vlvgf [[ELT:%v[0-9]+]], %r2, 0
; CHECK-CODE-DAG: larl [[REG:%r[0-5]]],
; CHECK-CODE-DAG: vl [[MASK:%v[0-9]+]], 0([[REG]])
; CHECK-CODE: vperm %v24, %v24, [[ELT]], [[MASK]]
; CHECK-CODE: br %r14

; CHECK-VECTOR: .byte 12
; CHECK-VECTOR-NEXT: .byte 13
; CHECK-VECTOR-NEXT: .byte 14
; CHECK-VECTOR-NEXT: .byte 15
; CHECK-VECTOR-NEXT: .byte 8
; CHECK-VECTOR-NEXT: .byte 9
; CHECK-VECTOR-NEXT: .byte 10
; CHECK-VECTOR-NEXT: .byte 11
; CHECK-VECTOR-NEXT: .byte 4
; CHECK-VECTOR-NEXT: .byte 5
; CHECK-VECTOR-NEXT: .byte 6
; CHECK-VECTOR-NEXT: .byte 7
; CHECK-VECTOR-NEXT: .byte 16
; CHECK-VECTOR-NEXT: .byte 17
; CHECK-VECTOR-NEXT: .byte 18
; CHECK-VECTOR-NEXT: .byte 19

  %elt0 = extractelement <4 x i32> %x, i32 3
  %elt1 = extractelement <4 x i32> %x, i32 2
  %elt2 = extractelement <4 x i32> %x, i32 1
  %elt3 = trunc i64 %y to i32
  %vec0 = insertelement <4 x i32> undef, i32 %elt0, i32 0
  %vec1 = insertelement <4 x i32> %vec0, i32 %elt1, i32 1
  %vec2 = insertelement <4 x i32> %vec1, i32 %elt2, i32 2
  %vec3 = insertelement <4 x i32> %vec2, i32 %elt3, i32 3
  ret <4 x i32> %vec3
}

