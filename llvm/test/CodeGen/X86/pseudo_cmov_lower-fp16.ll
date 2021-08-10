; RUN: llc < %s -mtriple=i386-linux-gnu -mattr=+avx512fp16 -mattr=+avx512vl -o - | FileCheck %s

; This test checks that only a single jne gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
define dso_local <32 x half> @foo3(<32 x half> %a, <32 x half> %b, i1 zeroext %sign) local_unnamed_addr #0 {
; CHECK-LABEL: foo3:
; CHECK: jne
; CHECK-NOT: jne
entry:
  %spec.select = select i1 %sign, <32 x half> %a, <32 x half> %b
  ret <32 x half> %spec.select
}

; This test checks that only a single jne gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
define dso_local <16 x half> @foo4(<16 x half> %a, <16 x half> %b, i1 zeroext %sign) local_unnamed_addr #0 {
; CHECK-LABEL: foo4:
; CHECK: jne
; CHECK-NOT: jne
entry:
  %spec.select = select i1 %sign, <16 x half> %a, <16 x half> %b
  ret <16 x half> %spec.select
}

; This test checks that only a single jne gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
define dso_local <8 x half> @foo5(<8 x half> %a, <8 x half> %b, i1 zeroext %sign) local_unnamed_addr #0 {
; CHECK-LABEL: foo5:
; CHECK: jne
; CHECK-NOT: jne
entry:
  %spec.select = select i1 %sign, <8 x half> %a, <8 x half> %b
  ret <8 x half> %spec.select
}
