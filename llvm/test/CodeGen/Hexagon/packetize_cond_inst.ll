; RUN: llc -mtriple=hexagon -tail-dup-size=1 < %s | FileCheck %s

target triple = "hexagon-unknown--elf"

; Make sure we put the two conditionally executed adds in a packet.
;     {
;       p0 = cmp.gt(r2, r1)
;       if (!p0.new) r0 = add(r2, r1)
;       if (p0.new) r0 = add(r0, #10)
;     }
; CHECK: cmp
; CHECK-NEXT: add
; CHECK-NEXT: add
define i32 @f0(i32 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a2, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = add nsw i32 %a0, 10
  br label %b3

b2:                                               ; preds = %b0
  %v2 = add nsw i32 %a2, %a1
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v3 = phi i32 [ %v1, %b1 ], [ %v2, %b2 ]
  %v4 = add nsw i32 %v3, 1
  ret i32 %v4
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv5" }
