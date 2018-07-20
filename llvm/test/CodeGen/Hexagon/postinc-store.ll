; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s

; Check that post-increment store instructions are being generated.
; CHECK: memw(r{{[0-9]+}}++#4) = r{{[0-9]+}}

define i32 @sum(i32* nocapture %a, i16* nocapture %b, i32 %n) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %lsr.iv = phi i32 [ %lsr.iv.next, %for.body ], [ 10, %entry ]
  %arrayidx.phi = phi i32* [ %a, %entry ], [ %arrayidx.inc, %for.body ]
  %arrayidx1.phi = phi i16* [ %b, %entry ], [ %arrayidx1.inc, %for.body ]
  %0 = load i32, i32* %arrayidx.phi, align 4
  %1 = load i16, i16* %arrayidx1.phi, align 2
  %conv = sext i16 %1 to i32
  %factor = mul i32 %0, 2
  %add3 = add i32 %factor, %conv
  store i32 %add3, i32* %arrayidx.phi, align 4

  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  %arrayidx1.inc = getelementptr i16, i16* %arrayidx1.phi, i32 1
  %lsr.iv.next = add i32 %lsr.iv, -1
  %exitcond = icmp eq i32 %lsr.iv.next, 0
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}
