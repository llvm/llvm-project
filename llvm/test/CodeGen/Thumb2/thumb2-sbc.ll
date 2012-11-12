; RUN: llc -march=thumb -mattr=+thumb2 < %s | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
; CHECK: f1
; CHECK: subs r0, r0, r2
    %tmp = sub i64 %a, %b
    ret i64 %tmp
}

; 734439407618 = 0x000000ab00000002
define i64 @f2(i64 %a) {
; CHECK: f2
; CHECK: subs r0, #2
; CHECK: sbc r1, r1, #171
    %tmp = sub i64 %a, 734439407618
    ret i64 %tmp
}

; 5066626890203138 = 0x0012001200000002
define i64 @f3(i64 %a) {
; CHECK: f3
; CHECK: subs  r0, #2
; CHECK: sbc r1, r1, #1179666
    %tmp = sub i64 %a, 5066626890203138
    ret i64 %tmp
}

; 3747052064576897026 = 0x3400340000000002
define i64 @f4(i64 %a) {
; CHECK: f4
; CHECK: subs  r0, #2
; CHECK: sbc r1, r1, #872428544
    %tmp = sub i64 %a, 3747052064576897026
    ret i64 %tmp
}

; 6221254862626095106 = 0x5656565600000002
define i64 @f5(i64 %a) {
; CHECK: f5
; CHECK: subs  r0, #2
; CHECK: adc r1, r1, #-1448498775
    %tmp = sub i64 %a, 6221254862626095106
    ret i64 %tmp
}

; 287104476244869122 = 0x03fc000000000002
define i64 @f6(i64 %a) {
; CHECK: f6
; CHECK: subs  r0, #2
; CHECK: sbc r1, r1, #66846720
    %tmp = sub i64 %a, 287104476244869122
    ret i64 %tmp
}

; Example from numerics code that manually computes wider-than-64 values.
;
; CHECK: livecarry:
; CHECK: adds
; CHECK: adc
define i64 @livecarry(i64 %carry, i32 %digit) nounwind {
  %ch = lshr i64 %carry, 32
  %cl = and i64 %carry, 4294967295
  %truncdigit = zext i32 %digit to i64
  %prod = add i64 %cl, %truncdigit
  %ph = lshr i64 %prod, 32
  %carryresult = add i64 %ch, %ph
  ret i64 %carryresult
}
