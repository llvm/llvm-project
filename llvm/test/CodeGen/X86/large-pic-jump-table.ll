; RUN: llc -code-model=large -relocation-model=pic %s -o - | FileCheck %s

target triple = "x86_64-linux-gnu"

define i32 @f(i32 %i) {
bb:
  switch i32 %i, label %bb4 [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
  ]

bb0:
  ret i32 1

bb1:
  ret i32 100

bb2:
  ret i32 200

bb3:
  ret i32 400

bb4:
  ret i32 300
}

; CHECK:      movabsq $.LJTI0_0@GOTOFF, [[R1:%r[a-z]{2}]]
; CHECK-NEXT: addq    [[R1]], [[R2:%r[a-z]{2}]]
; CHECK-NEXT: addq    ([[R2]],[[R3:%r[a-z]{2}]],8), [[R2]]
; CHECK-NEXT: jmpq    *[[R2]]

; CHECK: .LJTI0_0:
; CHECK-NEXT: .quad   .LBB0_2-.LJTI0_0
; CHECK-NEXT: .quad   .LBB0_3-.LJTI0_0
; CHECK-NEXT: .quad   .LBB0_4-.LJTI0_0
; CHECK-NEXT: .quad   .LBB0_5-.LJTI0_0
