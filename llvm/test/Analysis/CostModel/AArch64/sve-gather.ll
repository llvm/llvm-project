; Check getIntrinsicInstrCost in BasicTTIImpl.h for masked gather

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s 2>%t | FileCheck %s

; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define <vscale x 4 x i32> @masked_gather_nxv4i32(<vscale x 4 x i32*> %ld, <vscale x 4 x i1> %masks, <vscale x 4 x i32> %passthru) {
; CHECK-LABEL: 'masked_gather_nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 64 for instruction:   %res = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> %ld, i32 0, <vscale x 4 x i1> %masks, <vscale x 4 x i32> %passthru)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 4 x i32> %res
  %res = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32(<vscale x 4 x i32*> %ld, i32 0, <vscale x 4 x i1> %masks, <vscale x 4 x i32> %passthru)
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i32> @masked_gather_nxv8i32(<vscale x 8 x i32*> %ld, <vscale x 8 x i1> %masks, <vscale x 8 x i32> %passthru) {
; CHECK-LABEL: 'masked_gather_nxv8i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 128 for instruction:   %res = call <vscale x 8 x i32> @llvm.masked.gather.nxv8i32.nxv8p0i32(<vscale x 8 x i32*> %ld, i32 0, <vscale x 8 x i1> %masks, <vscale x 8 x i32> %passthru)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 8 x i32> %res
  %res = call <vscale x 8 x i32> @llvm.masked.gather.nxv8i32(<vscale x 8 x i32*> %ld, i32 0, <vscale x 8 x i1> %masks, <vscale x 8 x i32> %passthru)
  ret <vscale x 8 x i32> %res
}

define <4 x i32> @masked_gather_v4i32(<4 x i32*> %ld, <4 x i1> %masks, <4 x i32> %passthru) {
; CHECK-LABEL: 'masked_gather_v4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 29 for instruction:   %res = call <4 x i32> @llvm.masked.gather.v4i32.v4p0i32(<4 x i32*> %ld, i32 0, <4 x i1> %masks, <4 x i32> %passthru)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction: ret <4 x i32> %res

  %res = call <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> %ld, i32 0, <4 x i1> %masks, <4 x i32> %passthru)
  ret <4 x i32> %res
}

; Check it properly falls back to BasicTTIImpl when legalized MVT is not a vector
define <1 x i128> @masked_gather_v1i128(<1 x i128*> %ld, <1 x i1> %masks, <1 x i128> %passthru) {
; CHECK-LABEL: 'masked_gather_v1i128'
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %res = call <1 x i128> @llvm.masked.gather.v1i128.v1p0i128(<1 x i128*> %ld, i32 0, <1 x i1> %masks, <1 x i128> %passthru)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret <1 x i128> %res

  %res = call <1 x i128> @llvm.masked.gather.v1i128.v1p0i128(<1 x i128*> %ld, i32 0, <1 x i1> %masks, <1 x i128> %passthru)
  ret <1 x i128> %res
}

declare <vscale x 4 x i32> @llvm.masked.gather.nxv4i32(<vscale x 4 x i32*> %ptrs, i32 %align, <vscale x 4 x i1> %masks, <vscale x 4 x i32> %passthru)
declare <vscale x 8 x i32> @llvm.masked.gather.nxv8i32(<vscale x 8 x i32*> %ptrs, i32 %align, <vscale x 8 x i1> %masks, <vscale x 8 x i32> %passthru)
declare <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> %ptrs, i32 %align, <4 x i1> %masks, <4 x i32> %passthru)
declare <1 x i128> @llvm.masked.gather.v1i128.v1p0i128(<1 x i128*>, i32, <1 x i1>, <1 x i128>)

