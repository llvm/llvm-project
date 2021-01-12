; Check getIntrinsicInstrCost in BasicTTIImpl.h with for  masked scatter

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s 2>%t | FileCheck %s

; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning


define void @masked_scatter_nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32*> %ptrs, <vscale x 4 x i1> %masks) {
; CHECK-LABEL: 'masked_scatter_nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 64 for instruction:   call void @llvm.masked.scatter.nxv4i32.nxv4p0i32(<vscale x 4 x i32> %data, <vscale x 4 x i32*> %ptrs, i32 0, <vscale x 4 x i1> %masks)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret void

  call void @llvm.masked.scatter.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32*> %ptrs, i32 0, <vscale x 4 x i1> %masks)
  ret void
}

define void @masked_scatter_nxv8i32(<vscale x 8 x i32> %data, <vscale x 8 x i32*> %ptrs, <vscale x 8 x i1> %masks) {
; CHECK-LABEL: 'masked_scatter_nxv8i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 128 for instruction:   call void @llvm.masked.scatter.nxv8i32.nxv8p0i32(<vscale x 8 x i32> %data, <vscale x 8 x i32*> %ptrs, i32 0, <vscale x 8 x i1> %masks)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret void

  call void @llvm.masked.scatter.nxv8i32(<vscale x 8 x i32> %data, <vscale x 8 x i32*> %ptrs, i32 0, <vscale x 8 x i1> %masks)
  ret void
}

define void @masked_scatter_v4i32(<4 x i32> %data, <4 x i32*> %ptrs, <4 x i1> %masks) {
; CHECK-LABEL: 'masked_scatter_v4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 29 for instruction:   call void @llvm.masked.scatter.v4i32.v4p0i32(<4 x i32> %data, <4 x i32*> %ptrs, i32 0, <4 x i1> %masks)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret void

  call void @llvm.masked.scatter.v4i32(<4 x i32> %data, <4 x i32*> %ptrs, i32 0, <4 x i1> %masks)
  ret void
}

; Check it properly falls back to BasicTTIImpl when legalized MVT is not a vector
define void @masked_scatter_v1i128(<1 x i128> %data, <1 x i128*> %ptrs, <1 x i1> %masks) {
; CHECK-LABEL: 'masked_scatter_v1i128'
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   call void @llvm.masked.scatter.v1i128.v1p0i128(<1 x i128> %data, <1 x i128*> %ptrs, i32 0, <1 x i1> %masks)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret void

  call void @llvm.masked.scatter.v1i128.v1p0i128(<1 x i128> %data, <1 x i128*> %ptrs, i32 0, <1 x i1> %masks)
  ret void
}

declare void @llvm.masked.scatter.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32*> %ptrs, i32 %align, <vscale x 4 x i1> %masks)
declare void @llvm.masked.scatter.nxv8i32(<vscale x 8 x i32> %data, <vscale x 8 x i32*> %ptrs, i32 %align, <vscale x 8 x i1> %masks)
declare void @llvm.masked.scatter.v4i32(<4 x i32> %data, <4 x i32*> %ptrs, i32 %align, <4 x i1> %masks)
declare void @llvm.masked.scatter.v1i128.v1p0i128(<1 x i128> %data, <1 x i128*> %ptrs, i32 %align, <1 x i1> %masks)
