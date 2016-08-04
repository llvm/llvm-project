; All of these should be codegen'd without loading immediates
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin | FileCheck %s

define i64 @add_ll(i64 %a, i64 %b) nounwind {
entry:
        %tmp.2 = add i64 %b, %a         ; <i64> [#uses=1]
        ret i64 %tmp.2
; CHECK-LABEL: add_ll:
; CHECK: addc r4, r6, r4
; CHECK: adde r3, r5, r3
; CHECK: blr
}

define i64 @add_l_5(i64 %a) nounwind {
entry:
        %tmp.1 = add i64 %a, 5          ; <i64> [#uses=1]
        ret i64 %tmp.1
; CHECK-LABEL: add_l_5:
; CHECK: addic r4, r4, 5
; CHECK: addze r3, r3
; CHECK: blr
}

define i64 @add_l_m5(i64 %a) nounwind {
entry:
        %tmp.1 = add i64 %a, -5         ; <i64> [#uses=1]
        ret i64 %tmp.1
; CHECK-LABEL: add_l_m5:
; CHECK: addic r4, r4, -5
; CHECK: addme r3, r3
; CHECK: blr
}

