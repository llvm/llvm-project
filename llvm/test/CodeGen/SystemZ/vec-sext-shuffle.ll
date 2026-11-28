; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s
;
; Test that sign-extending the result of a shuffle that takes its leading
; lanes from the second shuffle operand unpacks the second operand. The
; shuffle mask indexes the concatenation of both operands, so mask values
; that are at least the lane count of one operand select lanes of the second.
;
; Registers other than the incoming arguments (%v24, %v26) are matched with
; variables, so that the checks do not depend on register allocation.

; Lanes 0-3 are b[0..3] (mask offset 8), lanes 4-7 are a[4..7].
define void @fun0(<8 x i16> %a, <8 x i16> %b, ptr %dst) {
; CHECK-LABEL: fun0:
; CHECK-DAG: vuphh [[BLANES:%v[0-9]+]], %v26
; CHECK-DAG: vuplhw [[ALANES:%v[0-9]+]], %v24
; CHECK-DAG: vst [[BLANES]], 0(%r2), 4
; CHECK-DAG: vst [[ALANES]], 16(%r2), 4
; CHECK: br %r14
  %s = shufflevector <8 x i16> %a, <8 x i16> %b,
                     <8 x i32> <i32 8, i32 9, i32 10, i32 11,
                                i32 4, i32 5, i32 6, i32 7>
  %z = sext <8 x i16> %s to <8 x i32>
  store <8 x i32> %z, ptr %dst
  ret void
}

; Lanes 0-1 are b[0..1] (mask offset 4), lanes 2-3 are a[2..3].
define void @fun1(<4 x i32> %a, <4 x i32> %b, ptr %dst) {
; CHECK-LABEL: fun1:
; CHECK: vuphf [[BLANES:%v[0-9]+]], %v26
; CHECK: vst [[BLANES]], 0(%r2), 4
; CHECK: br %r14
  %s = shufflevector <4 x i32> %a, <4 x i32> %b,
                     <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  %z = sext <4 x i32> %s to <4 x i64>
  store <4 x i64> %z, ptr %dst
  ret void
}

; The reproducer from the bug report: lanes 0-1 are b[6..7] (mask offset 22).
; Only the unpack chain that produces the first result register is checked.
; It must start from %v26 (b), not %v24 (a).
define <16 x i64> @fun2(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: fun2:
; CHECK: vuphb [[HW:%v[0-9]+]], %v26
; CHECK: vuplhw [[W:%v[0-9]+]], [[HW]]
; CHECK: vuplf {{%v[0-9]+}}, [[W]]
; CHECK: br %r14
  %s = shufflevector <16 x i8> %a, <16 x i8> %b,
                     <16 x i32> <i32 22, i32 23, i32 1, i32 2,
                                 i32 3, i32 4, i32 5, i32 6,
                                 i32 7, i32 8, i32 9, i32 10,
                                 i32 11, i32 12, i32 13, i32 14>
  %z = sext <16 x i8> %s to <16 x i64>
  ret <16 x i64> %z
}

; Lanes 0-3 are a[4..7]. The first operand must still be used for the leading
; lanes when the offset is nonzero, but less than the lane count.
define void @fun3(<8 x i16> %a, <8 x i16> %b, ptr %dst) {
; CHECK-LABEL: fun3:
; CHECK-DAG: vuplhw [[ALANES:%v[0-9]+]], %v24
; CHECK-DAG: vuphh [[BLANES:%v[0-9]+]], %v26
; CHECK-DAG: vst [[ALANES]], 0(%r2), 4
; CHECK-DAG: vst [[BLANES]], 16(%r2), 4
; CHECK: br %r14
  %s = shufflevector <8 x i16> %a, <8 x i16> %b,
                     <8 x i32> <i32 4, i32 5, i32 6, i32 7,
                                i32 8, i32 9, i32 10, i32 11>
  %z = sext <8 x i16> %s to <8 x i32>
  store <8 x i32> %z, ptr %dst
  ret void
}
