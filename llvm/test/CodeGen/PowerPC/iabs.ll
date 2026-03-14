; REQUIRES: asserts
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -stats 2>&1 | \
; RUN:   grep "4 .*Number of machine instrs printed"

;; Integer absolute value, should produce something as good as:
;;      srawi r2, r3, 31
;;      add r3, r3, r2
;;      xor r3, r3, r2
;;      blr 
define i32 @test(i32 %a) {
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs
}

