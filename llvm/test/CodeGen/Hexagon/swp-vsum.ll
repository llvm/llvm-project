; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv5 -O3 < %s | FileCheck %s

; Simple vector total.
; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: add([[REG:r([0-9]+)]],r{{[0-9]+}})
; CHECK-NEXT: add(r{{[0-9]+}},#4)
; CHECK-NEXT: [[REG]] = memw(r{{[0-9]+}}+r{{[0-9]+}}<<#0)
; CHECK-NEXT: endloop0

define i32 @foo(i32* %a, i32 %n) {
entry:
  br label %for.body

for.body:
  %sum.02 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx.phi = phi i32* [ %a, %entry ], [ %arrayidx.inc, %for.body ]
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i32, i32* %arrayidx.phi, align 4
  %add = add nsw i32 %0, %sum.02
  %inc = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 10000
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %add
}
