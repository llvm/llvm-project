; RUN: llc -mcpu=pwr7 -mattr=-vsx < %s | FileCheck %s
; XFAIL: *

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define fastcc i64 @g1(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g1

; CHECK-LABEL: @g1
; CHECK-NOT: mr 3,
; CHECK: blr
}

define fastcc i64 @g2(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g2

; CHECK-LABEL: @g2
; CHECK: mr 3, 4
; CHECK-NEXT: blr
}

define fastcc i64 @g3(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g3

; CHECK-LABEL: @g3
; CHECK: mr 3, 5
; CHECK-NEXT: blr
}

define fastcc i64 @g4(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g4

; CHECK-LABEL: @g4
; CHECK: mr 3, 6
; CHECK-NEXT: blr
}

define fastcc i64 @g5(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g5

; CHECK-LABEL: @g5
; CHECK: mr 3, 7
; CHECK-NEXT: blr
}

define fastcc i64 @g6(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g6

; CHECK-LABEL: @g6
; CHECK: mr 3, 8
; CHECK-NEXT: blr
}

define fastcc i64 @g7(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g7

; CHECK-LABEL: @g7
; CHECK: mr 3, 9
; CHECK-NEXT: blr
}

define fastcc i64 @g8(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g8

; CHECK-LABEL: @g8
; CHECK: mr 3, 10
; CHECK-NEXT: blr
}

define fastcc i64 @g9(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g9

; CHECK-LABEL: @g9
; CHECK: ld 3, 48(1)
; CHECK-NEXT: blr
}

define fastcc i64 @g10(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g10

; CHECK-LABEL: @g10
; CHECK: ld 3, 56(1)
; CHECK-NEXT: blr
}

define fastcc i64 @g11(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret i64 %g11

; CHECK-LABEL: @g11
; CHECK: ld 3, 64(1)
; CHECK-NEXT: blr
}

define fastcc double @f1(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f1

; CHECK-LABEL: @f1
; CHECK-NOT: fmr 1,
; CHECK: blr
}

define fastcc double @f2(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f2

; CHECK-LABEL: @f2
; CHECK: fmr 1, 2
; CHECK-NEXT: blr
}

define fastcc double @f3(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f3

; CHECK-LABEL: @f3
; CHECK: fmr 1, 3
; CHECK-NEXT: blr
}

define fastcc double @f4(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f4

; CHECK-LABEL: @f4
; CHECK: fmr 1, 4
; CHECK-NEXT: blr
}

define fastcc double @f5(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f5

; CHECK-LABEL: @f5
; CHECK: fmr 1, 5
; CHECK-NEXT: blr
}

define fastcc double @f6(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f6

; CHECK-LABEL: @f6
; CHECK: fmr 1, 6
; CHECK-NEXT: blr
}

define fastcc double @f7(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f7

; CHECK-LABEL: @f7
; CHECK: fmr 1, 7
; CHECK-NEXT: blr
}

define fastcc double @f8(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f8

; CHECK-LABEL: @f8
; CHECK: fmr 1, 8
; CHECK-NEXT: blr
}

define fastcc double @f9(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f9

; CHECK-LABEL: @f9
; CHECK: fmr 1, 9
; CHECK-NEXT: blr
}

define fastcc double @f10(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f10

; CHECK-LABEL: @f10
; CHECK: fmr 1, 10
; CHECK-NEXT: blr
}

define fastcc double @f11(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f11

; CHECK-LABEL: @f11
; CHECK: fmr 1, 11
; CHECK-NEXT: blr
}

define fastcc double @f12(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f12

; CHECK-LABEL: @f12
; CHECK: fmr 1, 12
; CHECK-NEXT: blr
}

define fastcc double @f13(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f13

; CHECK-LABEL: @f13
; CHECK: fmr 1, 13
; CHECK-NEXT: blr
}

define fastcc double @f14(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f14

; CHECK-LABEL: @f14
; CHECK: lfd 1, 120(1)
; CHECK-NEXT: blr
}

define fastcc double @f15(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret double %f15

; CHECK-LABEL: @f15
; CHECK: lfd 1, 152(1)
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v1(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v1

; CHECK-LABEL: @v1
; CHECK-NOT: vor 2,
; CHECK: blr
}

define fastcc <4 x i32> @v2(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v2

; CHECK-LABEL: @v2
; CHECK: vor 2, 3, 3
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v3(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v3

; CHECK-LABEL: @v3
; CHECK: vor 2, 4, 4
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v4(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v4

; CHECK-LABEL: @v4
; CHECK: vor 2, 5, 5
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v5(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v5

; CHECK-LABEL: @v5
; CHECK: vor 2, 6, 6
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v6(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v6

; CHECK-LABEL: @v6
; CHECK: vor 2, 7, 7
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v7(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v7

; CHECK-LABEL: @v7
; CHECK: vor 2, 8, 8
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v8(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v8

; CHECK-LABEL: @v8
; CHECK: vor 2, 9, 9
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v9(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v9

; CHECK-LABEL: @v9
; CHECK: vor 2, 10, 10
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v10(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v10

; CHECK-LABEL: @v10
; CHECK: vor 2, 11, 11
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v11(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v11

; CHECK-LABEL: @v11
; CHECK: vor 2, 12, 12
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v12(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v12

; CHECK-LABEL: @v12
; CHECK: vor 2, 13, 13
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v13(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v13

; CHECK-LABEL: @v13
; CHECK: addi [[REG1:[0-9]+]], 1, 96
; CHECK-NEXT: lvx 2, 0, [[REG1]]
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v14(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v14

; CHECK-LABEL: @v14
; CHECK: addi [[REG1:[0-9]+]], 1, 128
; CHECK-NEXT: lvx 2, 0, [[REG1]]
; CHECK-NEXT: blr
}

define fastcc <4 x i32> @v15(i64 %g1, double %f1, <4 x i32> %v1, i64 %g2, double %f2, <4 x i32> %v2, i64 %g3, double %f3, <4 x i32> %v3, i64 %g4, double %f4, <4 x i32> %v4, i64 %g5, double %f5, <4 x i32> %v5, i64 %g6, double %f6, <4 x i32> %v6, i64 %g7, double %f7, <4 x i32> %v7, i64 %g8, double %f8, <4 x i32> %v8, i64 %g9, double %f9, <4 x i32> %v9, i64 %g10, double %f10, <4 x i32> %v10, i64 %g11, double %f11, <4 x i32> %v11, i64 %g12, double %f12, <4 x i32> %v12, i64 %g13, double %f13, <4 x i32> %v13, i64 %g14, double %f14, <4 x i32> %v14, i64 %g15, double %f15, <4 x i32> %v15, i64 %g16, double %f16, <4 x i32> %v16) #0 {
  ret <4 x i32> %v15

; CHECK-LABEL: @v15
; CHECK: addi [[REG1:[0-9]+]], 1, 160
; CHECK-NEXT: lvx 2, 0, [[REG1]]
; CHECK-NEXT: blr
}

define void @cg1(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg1
; CHECK-NOT: {{^[ \t]*}}mr 3,
; CHECK: blr
}

define void @cg2(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg2
; CHECK: mr 4, 3
; CHECK: blr
}

define void @cg3(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg3
; CHECK: mr 5, 3
; CHECK: blr
}

define void @cg4(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg4
; CHECK: mr 6, 3
; CHECK: blr
}

define void @cg5(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg5
; CHECK: mr 7, 3
; CHECK: blr
}

define void @cg6(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg6
; CHECK: mr 8, 3
; CHECK: blr
}

define void @cg7(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg7
; CHECK: mr 9, 3
; CHECK: blr
}

define void @cg8(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg8
; CHECK: mr 10, 3
; CHECK: blr
}

define void @cg9(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg9
; CHECK: mr [[REG1:[0-9]+]], 3
; CHECK: std [[REG1]], 48(1)
; CHECK: blr
}

define void @cg10(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg10
; CHECK: mr [[REG1:[0-9]+]], 3
; CHECK: std [[REG1]], 56(1)
; CHECK: blr
}

define void @cg11(i64 %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 %v, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cg11
; CHECK: mr [[REG1:[0-9]+]], 3
; CHECK: std [[REG1]], 64(1)
; CHECK: blr
}

define void @cf1(double %v) #0 {
  tail call fastcc i64 @g1(i64 0, double %v, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cf1
; CHECK-NOT: fmr 1,
; CHECK: blr
}

define void @cf2(double %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double %v, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cf2
; CHECK: fmr 2, 1
; CHECK: blr
}

define void @cf3(double %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double %v, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cf3
; CHECK: fmr 3, 1
; CHECK: blr
}

define void @cf4(double %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double %v, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cf4
; CHECK: fmr 4, 1
; CHECK: blr
}

define void @cf5(double %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double %v, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cf5
; CHECK: fmr 5, 1
; CHECK: blr
}

define void @cf14(double %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double %v, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cf14
; CHECK: stfd 1, 120(1)
; CHECK: blr
}

define void @cf15(double %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double %v, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cf15
; CHECK: stfd 1, 152(1)
; CHECK: blr
}

define void @cv2(<4 x i32> %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> %v, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cv2
; CHECK: vor 3, 2, 2
; CHECK: blr
}

define void @cv3(<4 x i32> %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> %v, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cv3
; CHECK: vor 4, 2, 2
; CHECK: blr
}

define void @cv13(<4 x i32> %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> %v, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cv13
; CHECK-DAG: li [[REG1:[0-9]+]], 96
; CHECK-DAG: vor [[REG2:[0-9]+]], 3, 3
; CHECK: stvx [[REG2]], 1, [[REG1]]
; CHECK: blr
}

define void @cv14(<4 x i32> %v) #0 {
  tail call fastcc i64 @g1(i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> %v, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0, double 0.0, <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  ret void

; CHECK-LABEL: @cv14
; CHECK-DAG: li [[REG1:[0-9]+]], 128
; CHECK-DAG: vor [[REG2:[0-9]+]], 3, 3
; CHECK: stvx [[REG2]], 1, [[REG1]]
; CHECK: blr
}

attributes #0 = { nounwind }

