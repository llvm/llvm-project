; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv5 -O3 < %s | FileCheck %s

; Multiply and accumulate
; CHECK: mpyi([[REG0:r([0-9]+)]], [[REG1:r([0-9]+)]])
; CHECK-NEXT: add(r{{[0-9]+}}, #4)
; CHECK-NEXT: [[REG0]] = memw(r{{[0-9]+}} + r{{[0-9]+}}<<#0)
; CHECK-NEXT: [[REG1]] = memw(r{{[0-9]+}} + r{{[0-9]+}}<<#0)
; CHECK-NEXT: endloop0

define i32 @foo(i32* %a, i32* %b, i32 %n) {
entry:
  br label %for.body

for.body:
  %sum.03 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx.phi = phi i32* [ %a, %entry ], [ %arrayidx.inc, %for.body ]
  %arrayidx1.phi = phi i32* [ %b, %entry ], [ %arrayidx1.inc, %for.body ]
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i32, i32* %arrayidx.phi, align 4
  %1 = load i32, i32* %arrayidx1.phi, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %sum.03
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, 10000
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  %arrayidx1.inc = getelementptr i32, i32* %arrayidx1.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %add
}

