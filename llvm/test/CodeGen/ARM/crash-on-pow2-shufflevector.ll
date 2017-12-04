; RUN: llc < %s -mtriple=armv7 | FileCheck %s
;
; Ensure that don't crash given a largeish power-of-two shufflevector index.

%struct.desc = type { i32, [7 x i32] }

define i32 @foo(%struct.desc* %descs, i32 %num, i32 %cw) local_unnamed_addr #0 {
; CHECK-LABEL: foo:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    mov r1, #32
; CHECK-NEXT:    vld1.32 {d16, d17}, [r0], r1
; CHECK-NEXT:    vld1.32 {d18, d19}, [r0]
; CHECK-NEXT:    vtrn.32 q8, q9
; CHECK-NEXT:    vadd.i32 d16, d16, d16
; CHECK-NEXT:    vmov.32 r0, d16[1]
; CHECK-NEXT:    bx lr
entry:
  %descs.vec = bitcast %struct.desc* %descs to <16 x i32>*
  %wide.vec = load <16 x i32>, <16 x i32>* %descs.vec, align 4
  %strided.vec = shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <2 x i32> <i32 0, i32 8>
  %bin.rdx20 = add <2 x i32> %strided.vec, %strided.vec
  %0 = extractelement <2 x i32> %bin.rdx20, i32 1
  ret i32 %0
}
