; RUN: llc -verify-machineinstrs < %s -march=ppc64

define void @__divtc3({ ppc_fp128, ppc_fp128 }* noalias sret %agg.result, ppc_fp128 %a, ppc_fp128 %b, ppc_fp128 %c, ppc_fp128 %d) nounwind {
entry:
        %imag59 = load ppc_fp128, ppc_fp128* null, align 8         ; <ppc_fp128> [#uses=1]
        %0 = fmul ppc_fp128 0xM00000000000000000000000000000000, %imag59         ; <ppc_fp128> [#uses=1]
        %1 = fmul ppc_fp128 0xM00000000000000000000000000000000, 0xM00000000000000000000000000000000             ; <ppc_fp128> [#uses=1]
        %2 = fadd ppc_fp128 %0, %1               ; <ppc_fp128> [#uses=1]
        store ppc_fp128 %2, ppc_fp128* null, align 16
        unreachable
}
