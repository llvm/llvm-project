; RUN: llc -march=thumb -mattr=+thumb2 < %s | FileCheck %s

; 171 = 0x000000ab
define i64 @f1(i64 %a) {
; CHECK: f1
; CHECK: subs  r0, #171
; CHECK: sbc r1, r1, #0
    %tmp = sub i64 %a, 171
    ret i64 %tmp
}

; 1179666 = 0x00120012
define i64 @f2(i64 %a) {
; CHECK: f2
; CHECK: subs.w  r0, r0, #1179666
; CHECK: sbc r1, r1, #0
    %tmp = sub i64 %a, 1179666
    ret i64 %tmp
}

; 872428544 = 0x34003400
define i64 @f3(i64 %a) {
; CHECK: f3
; CHECK: subs.w  r0, r0, #872428544
; CHECK: sbc r1, r1, #0
    %tmp = sub i64 %a, 872428544
    ret i64 %tmp
}

; 1448498774 = 0x56565656
define i64 @f4(i64 %a) {
; CHECK: f4
; CHECK: subs.w  r0, r0, #1448498774
; CHECK: sbc r1, r1, #0
    %tmp = sub i64 %a, 1448498774
    ret i64 %tmp
}

; 66846720 = 0x03fc0000
define i64 @f5(i64 %a) {
; CHECK: f5
; CHECK: subs.w  r0, r0, #66846720
; CHECK: sbc r1, r1, #0
    %tmp = sub i64 %a, 66846720
    ret i64 %tmp
}

; 734439407618 = 0x000000ab00000002
define i64 @f6(i64 %a) {
; CHECK: f6
; CHECK: subs r0, #2
; CHECK: sbc r1, r1, #171
   %tmp = sub i64 %a, 734439407618
   ret i64 %tmp
}
