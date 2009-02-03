; RUN: llvm-as < %s | llc -march=arm | \
; RUN:   grep {mov r0, #0} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep {flds.*\\\[} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep {fsts.*\\\[} | count 1

define float @f1(float %a) {
        ret float 0.000000e+00
}

define float @f2(float* %v, float %u) {
        %tmp = load float* %v           ; <float> [#uses=1]
        %tmp1 = add float %tmp, %u              ; <float> [#uses=1]
        ret float %tmp1
}

define void @f3(float %a, float %b, float* %v) {
        %tmp = add float %a, %b         ; <float> [#uses=1]
        store float %tmp, float* %v
        ret void
}
