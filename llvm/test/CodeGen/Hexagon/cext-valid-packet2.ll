; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that the packetizer generates valid packets with constant
; extended add and base+offset store instructions.

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}add(r{{[0-9]+}},{{ *}}##200000)
; CHECK-NEXT: memw(r{{[0-9]+}}{{ *}}+{{ *}}##12000){{ *}}={{ *}}r{{[0-9]+}}.new
; CHECK-NEXT: }

define void @test(i32* nocapture %a, i32* nocapture %b, i32 %c) nounwind {
entry:
  %0 = load i32, i32* %a, align 4
  %add1 = add nsw i32 %0, 200000
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i32 3000
  store i32 %add1, i32* %arrayidx2, align 4
  ret void
}
