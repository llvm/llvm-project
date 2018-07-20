; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  call void bitcast (void (...)* @f1 to void (i8*)*)(i8* blockaddress(@f0, %b1))
  br label %b1

b1:                                               ; preds = %b2, %b0
  ret void

b2:                                               ; No predecessors!
  indirectbr i8* undef, [label %b1]
}

declare void @f1(...)

attributes #0 = { nounwind }
