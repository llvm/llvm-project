; Checks getIntrinsicInstrCost in BasicTTIImpl.h with SVE for CTLZ and CCTZ

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s 2>%t | FileCheck %s

; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; Check for CTLZ

define void  @ctlz_nxv4i32(<vscale x 4 x i32> %A) {
; CHECK-LABEL: 'ctlz_nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %1 = tail call <vscale x 4 x i32> @llvm.ctlz.nxv4i32(<vscale x 4 x i32> %A, i1 true)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret void

  %1 = tail call <vscale x 4 x i32> @llvm.ctlz.nxv4i32(<vscale x 4 x i32> %A, i1 true)
  ret void
}

; Check for CCTZ

define void  @cttz_nxv4i32(<vscale x 4 x i32> %A) {
; CHECK-LABEL: 'cttz_nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %1 = tail call <vscale x 4 x i32> @llvm.cttz.nxv4i32(<vscale x 4 x i32> %A, i1 true)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret void

  %1 = tail call <vscale x 4 x i32> @llvm.cttz.nxv4i32(<vscale x 4 x i32> %A, i1 true)
  ret void
}

declare <vscale x 4 x i32> @llvm.ctlz.nxv4i32(<vscale x 4 x i32>, i1)
declare <vscale x 4 x i32> @llvm.cttz.nxv4i32(<vscale x 4 x i32>, i1)
