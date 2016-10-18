; RUN: llc -mtriple=arm -mattr=+neon %s -o - | FileCheck %s

; Check codegen for 64-bit icmp operations, which don't directly map to any
; instruction.

define <2 x i64> @vne(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vne:
;CHECK: vceq.i32
;CHECK-NEXT: vrev64.32
;CHECK-NEXT: vand
;CHECK-NEXT: vmvn
;CHECK-NEXT: vmov
;CHECK-NEXT: vmov
;CHECK-NEXT: mov pc, lr
      %tmp1 = load <2 x i64>, <2 x i64>* %A
      %tmp2 = load <2 x i64>, <2 x i64>* %B
      %tmp3 = icmp ne <2 x i64> %tmp1, %tmp2
      %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
      ret <2 x i64> %tmp4
}

define <2 x i64> @veq(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: veq:
;CHECK: vceq.i32
;CHECK-NEXT: vrev64.32
;CHECK-NEXT: vand
;CHECK-NEXT: vmov
;CHECK-NEXT: vmov
;CHECK-NEXT: mov pc, lr
    %tmp1 = load <2 x i64>, <2 x i64>* %A
    %tmp2 = load <2 x i64>, <2 x i64>* %B
    %tmp3 = icmp eq <2 x i64> %tmp1, %tmp2
    %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
    ret <2 x i64> %tmp4
}

; FIXME: We currently generate terrible code for this.
; (Atop < Btop) | ((ATop == BTop) & (ABottom < BBottom))
; would come out to roughly 6 instructions, but we currently
; scalarize it.
define <2 x i64> @vult(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vult:
;CHECK: subs
;CHECK: sbcs
;CHECK: subs
;CHECK: sbcs
    %tmp1 = load <2 x i64>, <2 x i64>* %A
    %tmp2 = load <2 x i64>, <2 x i64>* %B
    %tmp3 = icmp ult <2 x i64> %tmp1, %tmp2
    %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
    ret <2 x i64> %tmp4
}
