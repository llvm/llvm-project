; RUN: opt < %s -postdomtree -analyze | FileCheck %s
define internal void @f() {
entry:
  br i1 undef, label %bb35, label %bb3.i

bb3.i:
  br label %bb3.i

bb:
  br label %bb35

bb.i:
  br label %bb35

_float32_unpack.exit:
  br label %bb35

bb.i5:
  br label %bb35

_float32_unpack.exit8:
  br label %bb35

bb32.preheader:
  br label %bb35

bb3:
  br label %bb35

bb3.split.us:
  br label %bb35

bb.i4.us:
  br label %bb35

bb7.i.us:
  br label %bb35

bb.i4.us.backedge:
  br label %bb35

bb1.i.us:
  br label %bb35

bb6.i.us:
  br label %bb35

bb4.i.us:
  br label %bb35

bb8.i.us:
  br label %bb35

bb3.i.loopexit.us:
  br label %bb35

bb.nph21:
  br label %bb35

bb4:
  br label %bb35

bb5:
  br label %bb35

bb14.preheader:
  br label %bb35

bb.nph18:
  br label %bb35

bb8.us.preheader:
  br label %bb35

bb8.preheader:
  br label %bb35

bb8.us:
  br label %bb35

bb8:
  br label %bb35

bb15.loopexit:
  br label %bb35

bb15.loopexit2:
  br label %bb35

bb15:
  br label %bb35

bb16:
  br label %bb35

bb17.loopexit.split:
  br label %bb35

bb.nph14:
  br label %bb35

bb19:
  br label %bb35

bb20:
  br label %bb35

bb29.preheader:
  br label %bb35

bb.nph:
  br label %bb35

bb23.us.preheader:
  br label %bb35

bb23.preheader:
  br label %bb35

bb23.us:
  br label %bb35

bb23:
  br label %bb35

bb30.loopexit:
  br label %bb35

bb30.loopexit1:
  br label %bb35

bb30:
  br label %bb35

bb31:
  br label %bb35

bb35.loopexit:
  br label %bb35

bb35.loopexit3:
  br label %bb35

bb35:
  ret void
}
; CHECK: Inorder PostDominator Tree: 
; CHECK-NEXT:   [1]  <<exit node>> {0,97}
; CHECK-NEXT:     [2] %bb35 {1,92}
; CHECK-NEXT:       [3] %bb35.loopexit3 {2,3}
; CHECK-NEXT:       [3] %bb35.loopexit {4,5}
; CHECK-NEXT:       [3] %bb31 {6,7}
; CHECK-NEXT:       [3] %bb30 {8,9}
; CHECK-NEXT:       [3] %bb30.loopexit1 {10,11}
; CHECK-NEXT:       [3] %bb30.loopexit {12,13}
; CHECK-NEXT:       [3] %bb23 {14,15}
; CHECK-NEXT:       [3] %bb23.us {16,17}
; CHECK-NEXT:       [3] %bb23.preheader {18,19}
; CHECK-NEXT:       [3] %bb23.us.preheader {20,21}
; CHECK-NEXT:       [3] %bb.nph {22,23}
; CHECK-NEXT:       [3] %bb29.preheader {24,25}
; CHECK-NEXT:       [3] %bb20 {26,27}
; CHECK-NEXT:       [3] %bb19 {28,29}
; CHECK-NEXT:       [3] %bb.nph14 {30,31}
; CHECK-NEXT:       [3] %bb17.loopexit.split {32,33}
; CHECK-NEXT:       [3] %bb16 {34,35}
; CHECK-NEXT:       [3] %bb15 {36,37}
; CHECK-NEXT:       [3] %bb15.loopexit2 {38,39}
; CHECK-NEXT:       [3] %bb15.loopexit {40,41}
; CHECK-NEXT:       [3] %bb8 {42,43}
; CHECK-NEXT:       [3] %bb8.us {44,45}
; CHECK-NEXT:       [3] %bb8.preheader {46,47}
; CHECK-NEXT:       [3] %bb8.us.preheader {48,49}
; CHECK-NEXT:       [3] %bb.nph18 {50,51}
; CHECK-NEXT:       [3] %bb14.preheader {52,53}
; CHECK-NEXT:       [3] %bb5 {54,55}
; CHECK-NEXT:       [3] %bb4 {56,57}
; CHECK-NEXT:       [3] %bb.nph21 {58,59}
; CHECK-NEXT:       [3] %bb3.i.loopexit.us {60,61}
; CHECK-NEXT:       [3] %bb8.i.us {62,63}
; CHECK-NEXT:       [3] %bb4.i.us {64,65}
; CHECK-NEXT:       [3] %bb6.i.us {66,67}
; CHECK-NEXT:       [3] %bb1.i.us {68,69}
; CHECK-NEXT:       [3] %bb.i4.us.backedge {70,71}
; CHECK-NEXT:       [3] %bb7.i.us {72,73}
; CHECK-NEXT:       [3] %bb.i4.us {74,75}
; CHECK-NEXT:       [3] %bb3.split.us {76,77}
; CHECK-NEXT:       [3] %bb3 {78,79}
; CHECK-NEXT:       [3] %bb32.preheader {80,81}
; CHECK-NEXT:       [3] %_float32_unpack.exit8 {82,83}
; CHECK-NEXT:       [3] %bb.i5 {84,85}
; CHECK-NEXT:       [3] %_float32_unpack.exit {86,87}
; CHECK-NEXT:       [3] %bb.i {88,89}
; CHECK-NEXT:       [3] %bb {90,91}
; CHECK-NEXT:     [2] %entry {93,94}
; CHECK-NEXT:     [2] %bb3.i {95,96}
