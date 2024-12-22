; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that the packetizer generates valid packets with constant
; extended instructions.
; CHECK: {
; CHECK-NEXT: r{{[0-9]+}} = add(r{{[0-9]+}},##{{[0-9]+}})
; CHECK-NEXT: r{{[0-9]+}} = add(r{{[0-9]+}},##{{[0-9]+}})
; CHECK-NEXT: }

define i32 @check-packet1(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %add = add nsw i32 %a, 200000
  %add1 = add nsw i32 %b, 200001
  %add2 = add nsw i32 %c, 200002
  %cmp = icmp sgt i32 %add, %add1
  %b.addr.0 = select i1 %cmp, i32 %add1, i32 %add2
  ret i32 %b.addr.0
}
