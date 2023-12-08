; RUN: llc -march=hexagon < %s | FileCheck %s

define i32 @foo(i32 %a, i32 %b) nounwind readnone {
; CHECK: lsl
; CHECK: aslh
entry:
  %shl1 = shl i32 16, %a
  %shl2 = shl i32 %b, 16
  %ret = mul i32 %shl1, %shl2
  ret i32 %ret
}

define i32 @bar(i32 %a, i32 %b) nounwind readnone {
; CHECK: asrh
; CHECK: lsr
entry:
  %shl1 = ashr i32 16, %a
  %shl2 = ashr i32 %b, 16
  %ret = mul i32 %shl1, %shl2
  ret i32 %ret
}
