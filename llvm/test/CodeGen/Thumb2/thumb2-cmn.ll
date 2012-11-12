; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; These tests could be improved by 'movs r0, #0' being rematerialized below the
; test as 'mov.w r0, #0'.

define i1 @f1(i32 %a, i32 %b) {
    %nb = sub i32 0, %b
    %tmp = icmp ne i32 %a, %nb
    ret i1 %tmp
}
; CHECK: f1:
; CHECK: 	cmn	{{.*}}, r1

define i1 @f2(i32 %a, i32 %b) {
    %nb = sub i32 0, %b
    %tmp = icmp ne i32 %nb, %a
    ret i1 %tmp
}
; CHECK: f2:
; CHECK: 	cmn	{{.*}}, r1

define i1 @f3(i32 %a, i32 %b) {
    %nb = sub i32 0, %b
    %tmp = icmp eq i32 %a, %nb
    ret i1 %tmp
}
; CHECK: f3:
; CHECK: 	cmn	{{.*}}, r1

define i1 @f4(i32 %a, i32 %b) {
    %nb = sub i32 0, %b
    %tmp = icmp eq i32 %nb, %a
    ret i1 %tmp
}
; CHECK: f4:
; CHECK: 	cmn	{{.*}}, r1

define i1 @f5(i32 %a, i32 %b) {
    %tmp = shl i32 %b, 5
    %nb = sub i32 0, %tmp
    %tmp1 = icmp eq i32 %nb, %a
    ret i1 %tmp1
}
; CHECK: f5:
; CHECK: 	cmn.w	{{.*}}, r1, lsl #5

define i1 @f6(i32 %a, i32 %b) {
    %tmp = lshr i32 %b, 6
    %nb = sub i32 0, %tmp
    %tmp1 = icmp ne i32 %nb, %a
    ret i1 %tmp1
}
; CHECK: f6:
; CHECK: 	cmn.w	{{.*}}, r1, lsr #6

define i1 @f7(i32 %a, i32 %b) {
    %tmp = ashr i32 %b, 7
    %nb = sub i32 0, %tmp
    %tmp1 = icmp eq i32 %a, %nb
    ret i1 %tmp1
}
; CHECK: f7:
; CHECK: 	cmn.w	{{.*}}, r1, asr #7

define i1 @f8(i32 %a, i32 %b) {
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %nb = sub i32 0, %tmp
    %tmp1 = icmp ne i32 %a, %nb
    ret i1 %tmp1
}
; CHECK: f8:
; CHECK: 	cmn.w	{{.*}}, {{.*}}, ror #8


define void @f9(i32 %a, i32 %b) nounwind optsize {
  tail call void asm sideeffect "cmn.w     r0, r1", ""() nounwind, !srcloc !0
  ret void
}

!0 = metadata !{i32 81}

; CHECK: f9:
; CHECK: 	cmn.w	r0, r1
