; RUN: llvm-as < %s | llc -march=arm -mattr=vfp2 | not grep fmdrr
; RUN: llvm-as < %s | llc -march=arm -mattr=vfp2 | not grep fmrrd

; naive codegen for this is:
; _i:
;        fmdrr d0, r0, r1
;        fmrrd r0, r1, d0
;        bx lr

define i64 @test(double %X) {
        %Y = bitcast double %X to i64
        ret i64 %Y
}
