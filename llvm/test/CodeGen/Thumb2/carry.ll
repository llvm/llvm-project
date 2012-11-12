; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
entry:
; CHECK: f1:
; CHECK: subs r0, r0, r2
; CHECK: sbcs r1, r3
	%tmp = sub i64 %a, %b
	ret i64 %tmp
}

define i64 @f2(i64 %a, i64 %b) {
entry:
; CHECK: f2:
; CHECK: adds r0, r0, r0
; CHECK: adcs r1, r1
; CHECK: subs r0, r0, r2
; CHECK: sbcs r1, r3
        %tmp1 = shl i64 %a, 1
	%tmp2 = sub i64 %tmp1, %b
	ret i64 %tmp2
}

; rdar://12559385
define i64 @f3(i32 %vi) {
entry:
; CHECK: f3:
; CHECK: movw [[REG:r[0-9]+]], #36102
; CHECK: sbcs r{{[0-9]+}}, [[REG]]
    %v0 = zext i32 %vi to i64
    %v1 = xor i64 %v0, -155057456198619
    %v4 = add i64 %v1, 155057456198619
    %v5 = add i64 %v4, %v1
    ret i64 %v5
}
