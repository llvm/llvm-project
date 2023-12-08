; RUN: llc < %s -march=sparcv9 | FileCheck %s

target datalayout = "E-i64:64-n32:64-S128"
target triple = "sparc64-sun-sparc"

; CHECK-LABEL: test_and_spill
; CHECK:       and %i0, %i1, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_and_spill(i64 %a, i64 %b) {
entry:
  %r0 = and i64 %a, %b
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

; CHECK-LABEL: test_or_spill
; CHECK:       or %i0, %i1, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_or_spill(i64 %a, i64 %b) {
entry:
  %r0 = or i64 %a, %b
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

; CHECK-LABEL: test_xor_spill
; CHECK:       xor %i0, %i1, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_xor_spill(i64 %a, i64 %b) {
entry:
  %r0 = xor i64 %a, %b
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}


; CHECK-LABEL: test_add_spill
; CHECK:       add %i0, %i1, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_add_spill(i64 %a, i64 %b) {
entry:
  %r0 = add i64 %a, %b
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

; CHECK-LABEL: test_sub_spill
; CHECK:       sub %i0, %i1, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_sub_spill(i64 %a, i64 %b) {
entry:
  %r0 = sub i64 %a, %b
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

; CHECK-LABEL: test_andi_spill
; CHECK:       and %i0, 1729, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_andi_spill(i64 %a) {
entry:
  %r0 = and i64 %a, 1729
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

; CHECK-LABEL: test_ori_spill
; CHECK:       or %i0, 1729, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_ori_spill(i64 %a) {
entry:
  %r0 = or i64 %a, 1729
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

; CHECK-LABEL: test_xori_spill
; CHECK:       xor %i0, 1729, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_xori_spill(i64 %a) {
entry:
  %r0 = xor i64 %a, 1729
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

; CHECK-LABEL: test_addi_spill
; CHECK:       add %i0, 1729, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_addi_spill(i64 %a) {
entry:
  %r0 = add i64 %a, 1729
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

; CHECK-LABEL: test_subi_spill
; CHECK:       add %i0, -1729, [[R:%[gilo][0-7]]]
; CHECK:       stx [[R]], [%fp+{{.+}}]
; CHECK:       ldx [%fp+{{.+}}, %i0
define i64 @test_subi_spill(i64 %a) {
entry:
  %r0 = sub i64 %a, 1729
  %0 = tail call i64 asm sideeffect "#$0 $1", "=r,r,~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6}"(i64 %r0)
  ret i64 %r0
}

