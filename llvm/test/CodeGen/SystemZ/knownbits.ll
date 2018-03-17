; Test that DAGCombiner gets helped by computeKnownBitsForTargetNode().
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s  | FileCheck %s

; SystemZISD::REPLICATE
define i32 @f0() {
; CHECK-LABEL: f0:
; CHECK-LABEL: # %bb.0:
; CHECK:       vlgvf
; CHECK-NOT:   lhi %r2, 0
; CHECK-NOT:   chi %r0, 0
; CHECK-NOT:   lochilh %r2, 1
; CHECK: br %r14
  %cmp0 = icmp ne <4 x i32> undef, zeroinitializer
  %zxt0 = zext <4 x i1> %cmp0 to <4 x i32>
  %ext0 = extractelement <4 x i32> %zxt0, i32 3
  br label %exit

exit:
; The vector icmp+zext involves a REPLICATE of 1's. If KnownBits reflects
; this, DAGCombiner can see that the i32 icmp and zext here are not needed.
  %cmp1 = icmp ne i32 %ext0, 0
  %zxt1 = zext i1 %cmp1 to i32
  ret i32 %zxt1
}

; SystemZISD::JOIN_DWORDS (and REPLICATE)
define void @f1() {
; The DAG XOR has JOIN_DWORDS and REPLICATE operands. With KnownBits properly set
; for both these nodes, ICMP is used instead of TM during lowering because
; adjustForRedundantAnd() succeeds.
; CHECK-LABEL: f1:
; CHECK-LABEL: # %bb.0:
; CHECK-NOT:   tmll
; CHECK-NOT:   jne
; CHECK:       cijlh
  %1 = load i16, i16* null, align 2
  %2 = icmp eq i16 %1, 0
  %3 = insertelement <2 x i1> undef, i1 %2, i32 0
  %4 = insertelement <2 x i1> %3, i1 true, i32 1
  %5 = xor <2 x i1> %4, <i1 true, i1 true>
  %6 = extractelement <2 x i1> %5, i32 0
  %7 = or i1 %6, undef
  br i1 %7, label %9, label %8

; <label>:8:                                      ; preds = %0
  unreachable

; <label>:9:                                      ; preds = %0
  unreachable
}
