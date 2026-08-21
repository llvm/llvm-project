; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

; This test ensures that a library shader with a UAV does not get the module and
; shader feature flag UAVsAtEveryStage when the DXIL validator version is >= 1.8

target triple = "dxil-pc-shadermodel6.5-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00000000

; CHECK-NOT: Note: shader requires additional functionality:
; CHECK-NOT:        UAVs at every shader stage

; CHECK: Function test : 0x00000000
define void @test() "hlsl.export" {
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
