; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  call void @f1(ptr blockaddress(@f0, %b1))
  br label %b1

b1:                                               ; preds = %b2, %b0
  ret void

b2:                                               ; No predecessors!
  indirectbr ptr undef, [label %b1]
}

declare void @f1(...)

attributes #0 = { nounwind }
