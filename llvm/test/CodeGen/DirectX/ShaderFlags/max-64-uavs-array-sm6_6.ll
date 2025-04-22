; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

; This test makes sure that resource arrays sizes are accounted for when
; counting the number of UAVs for setting the shader flag '64 UAV slots' when
; the shader model version is >= 6.6

; Note: there is no feature flag here (only a module flag), so we don't have an
; object test.

target triple = "dxil-pc-shadermodel6.6-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00008000

; CHECK: Note: shader requires additional functionality:
; CHECK:        64 UAV slots

; CHECK: Function test : 0x00000000
define void @test() "hlsl.export" {
  ; RWBuffer<float> Buf : register(u0, space0)
  %buf0 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 0, i32 1, i32 0, i1 false)

  ; RWBuffer<float> Buf[8] : register(u1, space0)
  %buf1 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 1, i32 8, i32 0, i1 false)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.resmayalias", i32 1}
