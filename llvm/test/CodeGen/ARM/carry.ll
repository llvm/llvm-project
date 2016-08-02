; RUN: llc -mtriple=armv6t2-eabi %s -o - | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: subs r
; CHECK: sbc r
entry:
	%tmp = sub i64 %a, %b
	ret i64 %tmp
}

define i64 @f2(i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: adc r
; CHECK: subs r
; CHECK: sbc r
entry:
        %tmp1 = shl i64 %a, 1
	%tmp2 = sub i64 %tmp1, %b
	ret i64 %tmp2
}

; add with live carry
define i64 @f3(i32 %al, i32 %bl) {
; CHECK-LABEL: f3:
; CHECK: adds r
; CHECK: adc r
entry:
        ; unsigned wide add
        %aw = zext i32 %al to i64
        %bw = zext i32 %bl to i64
        %cw = add i64 %aw, %bw
        ; ch == carry bit
        %ch = lshr i64 %cw, 32
	%dw = add i64 %ch, %bw
	ret i64 %dw
}

; rdar://10073745
define i64 @f4(i64 %x) nounwind readnone {
entry:
; CHECK-LABEL: f4:
; CHECK: rsbs r
; CHECK: rsc r
  %0 = sub nsw i64 0, %x
  ret i64 %0
}

; rdar://12559385
define i64 @f5(i32 %vi) {
entry:
; CHECK-LABEL: f5:
; CHECK: movw [[REG:r[0-9]+]], #36102
; CHECK: sbc r{{[0-9]+}}, r{{[0-9]+}}, [[REG]]
    %v0 = zext i32 %vi to i64
    %v1 = xor i64 %v0, -155057456198619
    %v4 = add i64 %v1, 155057456198619
    %v5 = add i64 %v4, %v1
    ret i64 %v5
}
