; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s

@test1 = global [2 x i24] [i24 -1, i24 -1]
; CHECK-LABEL: test1:
; CHECK-NEXT: .short	65535
; CHECK-NEXT: .byte	255
; CHECK-NEXT: .space	1
; CHECK-NEXT: .short	65535
; CHECK-NEXT: .byte	255
; CHECK-NEXT: .space	1

@test2 = global [2 x i7] [i7 1, i7 1]
; CHECK-LABEL: test2:
; CHECK-NEXT: .space 2,1

@test3 = global [4 x i128] [i128 -1, i128 -1, i128 -1, i128 -1]
; CHECK-LABEL: test3:
; CHECK-NEXT: .space 64,255

@test4 = global [3 x i16] [i16 257, i16 257, i16 257]
; CHECK-LABEL: test4:
; CHECK-NEXT: .space 6,1

@test5 = global [2 x [2 x i16]] [[2 x i16] [i16 257, i16 257], [2 x i16] [i16 -1, i16 -1]]
; CHECK-LABEL: test5:
; CHECK-NEXT: .space 4,1
; CHECK-NEXT: .space 4,255

@test6 = global [2 x [2 x i16]] [[2 x i16] [i16 257, i16 257], [2 x i16] [i16 257, i16 257]]
; CHECK-LABEL: test6:
; CHECK-NEXT: .space 8,1
