; RUN: llc < %s -march=thumb -stats 2>&1 | \
; RUN:   grep "4 .*Number of machine instrs printed"

;; Integer absolute value, should produce something as good as:
;; Thumb:
;;   movs r0, r0
;;   bpl
;;   rsb r0, r0, #0 (with opitmization, bpl + rsb is if-converted into rsbmi)
;;   bx lr

define i32 @test(i32 %a) {
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs
; CHECK:  movs r0, r0
; CHECK:  bpl
; CHECK:  rsb r0, r0, #0
; CHECK:  bx lr
}


