; RUN: llc < %s -march=ppc64

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc64-apple-darwin9.2.0"

define i128 @__fixunstfti(ppc_fp128 %a) nounwind  {
entry:
        %tmp1213 = uitofp i128 0 to ppc_fp128           ; <ppc_fp128> [#uses=1]
        %tmp15 = fsub ppc_fp128 %a, %tmp1213             ; <ppc_fp128> [#uses=1]
        %tmp2829 = fptoui ppc_fp128 %tmp15 to i64               ; <i64> [#uses=1]
        %tmp282930 = zext i64 %tmp2829 to i128          ; <i128> [#uses=1]
        %tmp32 = add i128 %tmp282930, 0         ; <i128> [#uses=1]
        ret i128 %tmp32
}
