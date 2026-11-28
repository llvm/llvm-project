; RUN: llc -march=hexagon -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv75 -mattr=+hvxv75,+hvx-length128b < %s | FileCheck %s

; Check that at most one shift-resource-only instruction (3-source narrowing
; shift or prefixsum) is placed in a packet for >= v79. These instructions
; cannot use the shift/permute resource. This also applies for < v79, since
; only one shift resource is available.

; Check that two 3 src shifts are NOT packetized together.
; CHECK-LABEL: two_3SRC
; CHECK: {
; CHECK-NEXT: v{{[0-9]+}}.uh = vasr(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}.uh):rnd:sat
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: v{{[0-9]+}}.uh = vasr(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}.uh):rnd:sat
; CHECK-NEXT: }

define <32 x i32> @two_3SRC(<64 x i32> %a, <32 x i32> %b, <32 x i32> %c) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vasrvwuhrndsat.128B(<64 x i32> %a, <32 x i32> %b)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vasrvwuhrndsat.128B(<64 x i32> %a, <32 x i32> %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybv.128B(<32 x i32> %0, <32 x i32> %1)
  ret <32 x i32> %2
}

; Check that two prefixsum instructions are NOT packetized together.
; CHECK-LABEL: two_prefixsum
; CHECK: v{{[0-9]+}}.b = prefixsum(q{{[0-3]}})
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: v{{[0-9]+}}.b = prefixsum(q{{[0-3]}})
; CHECK-NEXT: }

define <32 x i32> @two_prefixsum(<32 x i32> %a, i32 %b, i32 %c) #0 {
entry:
  %0 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 %b)
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vprefixqb.128B(<128 x i1> %0)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vprefixqb.128B(<128 x i1> %1)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybv.128B(<32 x i32> %2, <32 x i32> %3)
  ret <32 x i32> %4
}

; Check that a 3 src shift and a prefixsum instruction are NOT packetized together.
; CHECK-LABEL: prefixsum_3SRC
; CHECK: v{{[0-9]+}}.uh = vasr(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}.uh):rnd:sat
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: v{{[0-9]+}}.b = prefixsum(q{{[0-3]}})
; CHECK-NEXT: }

define <32 x i32> @prefixsum_3SRC(<64 x i32> %a, <32 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %b, i32 %c)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vasrvwuhrndsat.128B(<64 x i32> %a, <32 x i32> %b)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vprefixqb.128B(<128 x i1> %0)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybv.128B(<32 x i32> %1, <32 x i32> %2)
  ret <32 x i32> %3
}

attributes #0 = { nounwind readnone }
