; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

; This test makes sure that the shader flag '64 UAV slots' is set when there are
; more than 8 UAVs in the module.

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00008000

; CHECK: Note: shader requires additional functionality:
; CHECK:       64 UAV slots

; Note: 64 UAV slots does not get set per-function
; CHECK: Function test : 0x00008000
define void @test() "hlsl.export" {
  ; RWBuffer<float> Buf : register(u0, space0)
  %buf0 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 0, i32 1, i32 0, ptr null)
  ; RWBuffer<float> Buf : register(u1, space0)
  %buf1 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 1, i32 1, i32 0, ptr null)
  ; RWBuffer<float> Buf : register(u2, space0)
  %buf2 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 2, i32 1, i32 0, ptr null)
  ; RWBuffer<float> Buf : register(u3, space0)
  %buf3 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 3, i32 1, i32 0, ptr null)
  ; RWBuffer<float> Buf : register(u4, space0)
  %buf4 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 4, i32 1, i32 0, ptr null)
  ; RWBuffer<float> Buf : register(u5, space0)
  %buf5 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 5, i32 1, i32 0, ptr null)
  ; RWBuffer<float> Buf : register(u6, space0)
  %buf6 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 6, i32 1, i32 0, ptr null)
  ; RWBuffer<float> Buf : register(u7, space0)
  %buf7 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 7, i32 1, i32 0, ptr null)
  ; RWBuffer<float> Buf : register(u8, space0)
  %buf8 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 8, i32 1, i32 0, ptr null)
  ret void
}

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!0 = !{i32 1, !"dx.resmayalias", i32 1}
!1 = !{i32 1, i32 8}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:       Max64UAVs:         true
; DXC:       NextUnusedBit:   false
; DXC: ...
