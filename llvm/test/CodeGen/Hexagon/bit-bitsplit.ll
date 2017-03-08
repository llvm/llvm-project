; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: bitsplit(r{{[0-9]+}},#5)

target triple = "hexagon"

define i32 @fred(i32 %a, i32* nocapture readonly %b) local_unnamed_addr #0 {
entry:
  %and = and i32 %a, 31
  %shr = lshr i32 %a, 5
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %shr
  %0 = load i32, i32* %arrayidx, align 4
  %shr1 = lshr i32 %0, %and
  %and2 = and i32 %shr1, 1
  ret i32 %and2
}

attributes #0 = { norecurse nounwind readonly "target-cpu"="hexagonv60" "target-features"="-hvx,-hvx-double" }
