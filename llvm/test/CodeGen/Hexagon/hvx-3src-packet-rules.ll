; RUN: llc -march=hexagon -mcpu=hexagonv75 -mattr=+hvxv75,+hvx-length128b < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b < %s | FileCheck %s

; Check that all packet rules for 3 source shift are respected by the compiler
; * This instruction cannot be paired with a HVX permute instruction
; * If a packet contains this instruction and a HVX ALU op, then the ALU OP must be unary

; Check that a 3 src shift and a HVX permute are NOT packetized together
; CHECK-LABEL: CVI_VS_3SRC_AND_CVI_VP
; CHECK: {
; CHECK-NEXT: v{{[0-9]+}}.uh = vasr(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}.uh):rnd:sat
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: v{{[0-9]+}} = vdelta(v{{[0-9]+}},v{{[0-9]+}})
; CHECK-NEXT: }

define <32 x i32> @CVI_VS_3SRC_AND_CVI_VP(<64 x i32> %a, <32 x i32> %b, <32 x i32> %c) {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vasrvwuhrndsat.128B(<64 x i32> %a, <32 x i32> %b)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vdelta.128B(<32 x i32> %b, <32 x i32> %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybv.128B(<32 x i32> %0, <32 x i32> %1)
  ret <32 x i32> %2
}

; Check that a 3 src shift and a 2 source HVX ALU are NOT packetized together
; CHECK-LABEL: CVI_VS_3SRC_AND_CVI_VA_2SRC
; CHECK: {
; CHECK-NEXT: v{{[0-9]+}}.uh = vasr(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}.uh):rnd:sat
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: v{{[0-9]+}}.w = vadd(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: }

define <32 x i32> @CVI_VS_3SRC_AND_CVI_VA_2SRC(<64 x i32> %a, <32 x i32> %b, <32 x i32> %c) {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vasrvwuhrndsat.128B(<64 x i32> %a, <32 x i32> %b)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %b, <32 x i32> %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybv.128B(<32 x i32> %0, <32 x i32> %1)
  ret <32 x i32> %2
}
