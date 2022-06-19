; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define i64 @cmpxchg_0(i64* %ptr, i64 %compare, i64 %new_value) {
; CHECK-LABEL: @cmpxchg_0(
; CHECK-NEXT:    %tmp0 = cmpxchg i64* %ptr, i64 %compare, i64 %new_value seq_cst seq_cst
; CHECK-NEXT:    %tmp2 = extractvalue { i64, i1 } %tmp0, 0
; CHECK-NEXT:    ret i64 %tmp2
;
  %tmp0 = cmpxchg i64* %ptr, i64 %compare, i64 %new_value seq_cst seq_cst
  %tmp1 = extractvalue { i64, i1 } %tmp0, 1
  %tmp2 = extractvalue { i64, i1 } %tmp0, 0
  %tmp3 = select i1 %tmp1, i64 %compare, i64 %tmp2
  ret i64 %tmp3
}

define i64 @cmpxchg_1(i64* %ptr, i64 %compare, i64 %new_value) {
; CHECK-LABEL: @cmpxchg_1(
; CHECK-NEXT:    %tmp0 = cmpxchg i64* %ptr, i64 %compare, i64 %new_value seq_cst seq_cst
; CHECK-NEXT:    ret i64 %compare
;
  %tmp0 = cmpxchg i64* %ptr, i64 %compare, i64 %new_value seq_cst seq_cst
  %tmp1 = extractvalue { i64, i1 } %tmp0, 1
  %tmp2 = extractvalue { i64, i1 } %tmp0, 0
  %tmp3 = select i1 %tmp1, i64 %tmp2, i64 %compare
  ret i64 %tmp3
}

define i64 @cmpxchg_2(i64* %ptr, i64 %compare, i64 %new_value) {
; CHECK-LABEL: @cmpxchg_2(
; CHECK-NEXT:    %tmp0 = cmpxchg i64* %ptr, i64 %compare, i64 %new_value acq_rel monotonic
; CHECK-NEXT:    ret i64 %compare
;
  %tmp0 = cmpxchg i64* %ptr, i64 %compare, i64 %new_value acq_rel monotonic
  %tmp1 = extractvalue { i64, i1 } %tmp0, 1
  %tmp2 = extractvalue { i64, i1 } %tmp0, 0
  %tmp3 = select i1 %tmp1, i64 %compare, i64 %tmp2
  %tmp4 = select i1 %tmp1, i64 %tmp3, i64 %compare
  ret i64 %tmp4
}
