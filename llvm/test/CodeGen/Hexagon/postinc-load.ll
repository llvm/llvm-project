; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s

; Check that post-increment load instructions are being generated.
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memw(r{{[0-9]+}}{{ *}}++{{ *}}#4{{ *}})

define i32 @sum(i32* nocapture %a, i16* nocapture %b, i32 %n) nounwind {
entry:
  br label %for.body

for.body:
  %lsr.iv = phi i32 [ %lsr.iv.next, %for.body ], [ 10, %entry ]
  %arrayidx.phi = phi i32* [ %a, %entry ], [ %arrayidx.inc, %for.body ]
  %arrayidx1.phi = phi i16* [ %b, %entry ], [ %arrayidx1.inc, %for.body ]
  %sum.03 = phi i32 [ 0, %entry ], [ %add2, %for.body ]
  %0 = load i32, i32* %arrayidx.phi, align 4
  %1 = load i16, i16* %arrayidx1.phi, align 2
  %conv = sext i16 %1 to i32
  %add = add i32 %0, %sum.03
  %add2 = add i32 %add, %conv
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  %arrayidx1.inc = getelementptr i16, i16* %arrayidx1.phi, i32 1
  %lsr.iv.next = add i32 %lsr.iv, -1
  %exitcond = icmp eq i32 %lsr.iv.next, 0
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %add2
}

