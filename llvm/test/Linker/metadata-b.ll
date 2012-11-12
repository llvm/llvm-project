; This file is for use with metadata-a.ll
; RUN: true

define void @goo(i32 %b) nounwind {
entry:
  ret void, !attach !0, !and !{ i32 %b }
}

!0 = metadata !{i32 524334, void (i32)* @goo}
