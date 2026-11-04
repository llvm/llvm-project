; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; TODO: Remove this comment and add 'RUN' to the line below once vertex shaders are supported by llc
; llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

; This test ensures that a Vertex shader with a UAV gets the module and
; shader feature flag UAVsAtEveryStage

target triple = "dxil-pc-shadermodel6.5-vertex"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00010000

; CHECK: Note: shader requires additional functionality:
; CHECK:        UAVs at every shader stage

; CHECK: Function VSMain : 0x00010000
define void @VSMain() {
  ; RWBuffer<float> Buf : register(u0, space0)
  %buf0 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 0, i32 1, i32 0, ptr null)
  ret void
}

!dx.valver = !{!1}
!1 = !{i32 1, i32 8}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.resmayalias", i32 1}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:       UAVsAtEveryStage:         true
; DXC:       NextUnusedBit:   false
; DXC: ...
