; Test that DAGCombiner gets helped by getKnownBitsForTargetNode() when
; BITCAST nodes are involved on a big-endian target.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s  | FileCheck %s

define void @fun() {
entry:
  br label %lab0

lab0:
  %phi = phi i64 [ %sel, %lab0 ], [ 0, %entry ]
  %add = add nuw nsw i64 %phi, 1
  %add2 = add nuw nsw i64 %phi, 2
  %cmp = icmp eq i64 %add, undef
  %cmp2 = icmp eq i64 %add2, undef
  %ins = insertelement <2 x i1> undef, i1 %cmp, i32 0
  %ins2 = insertelement <2 x i1> undef, i1 %cmp2, i32 0
  %xor = xor <2 x i1> %ins, %ins2
  %extr = extractelement <2 x i1> %xor, i32 0
; The EXTRACT_VECTOR_ELT is done first into an i32, and then AND:ed with
; 1. The AND is not actually necessary since the element contains a CC (i1)
; value. Test that the BITCAST nodes in the DAG when computing KnownBits is
; handled so that the AND is removed. If this succeeds, this results in a CHI
; instead of TMLL.

; CHECK-LABEL: # %bb.0:
; CHECK:       chi
; CHECK-NOT:   tmll
; CHECK:       j
  %sel = select i1 %extr, i64 %add, i64 0
  br label %lab0
}
