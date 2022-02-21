; RUN: llc -mtriple=arm-eabi -mattr=+v4t %s -o - | FileCheck %s

;; Integer absolute value, should produce something as good as: ARM:
;;   movs r0, r0
;;   rsbmi r0, r0, #0
;;   bx lr

define i32 @test(i32 %a) {
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs
; CHECK:  cmp
; CHECK:  rsbmi r0, r0, #0
; CHECK:  bx lr
}

; rdar://11633193
;; 3 instructions will be generated for abs(a-b):
;;   subs
;;   rsbmi
;;   bx
define i32 @test2(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK: test2
; CHECK: subs
; CHECK-NEXT: rsbmi
; CHECK-NEXT: bx
  %sub = sub nsw i32 %a, %b
  %cmp = icmp sgt i32 %sub, -1
  %sub1 = sub nsw i32 0, %sub
  %cond = select i1 %cmp, i32 %sub, i32 %sub1
  ret i32 %cond
}

define i64 @test3(i64 %a) {
; CHECK-LABEL: test3:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    adds r0, r0, r1, asr #31
; CHECK-NEXT:    adc r2, r1, r1, asr #31
; CHECK-NEXT:    eor r0, r0, r1, asr #31
; CHECK-NEXT:    eor r1, r2, r1, asr #31
; CHECK-NEXT:    bx lr
  %tmp1neg = sub i64 0, %a
  %b = icmp sgt i64 %a, -1
  %abs = select i1 %b, i64 %a, i64 %tmp1neg
  ret i64 %abs
}
