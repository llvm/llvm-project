; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Verify that the legacy WMMA IU8 intrinsic without the clamp operand is
; upgraded by appending clamp=false.

define <8 x i32> @wmma_legacy(<8 x i32> %a, <8 x i32> %b, <8 x i32> %c) {
; CHECK-LABEL: @wmma_legacy(
; CHECK-NEXT: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32(i1 false, <8 x i32> %a, i1 false, <8 x i32> %b, <8 x i32> %c, i1 false, i1 false, i1 false) #1, !annotation !0
; CHECK-NEXT: ret <8 x i32>
  %res = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32(
      i1 false, <8 x i32> %a, i1 false, <8 x i32> %b, <8 x i32> %c,
      i1 false, i1 false) #1, !annotation !0
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32(
  i1, <8 x i32>, i1, <8 x i32>, <8 x i32>, i1, i1)

attributes #1 = { cold }

!0 = !{!"wmma-upgrade"}
