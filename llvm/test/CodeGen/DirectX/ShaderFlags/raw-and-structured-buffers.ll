; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

; Note: there is no feature flag here (only a module flag), so we don't have an
; object test.

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00000010

; CHECK: Note: shader requires additional functionality:
; CHECK:       Raw and structured buffers

; CHECK: Function rawbuf : 0x00000010
define float @rawbuf() "hlsl.export" {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %load = call {float, i1} @llvm.dx.resource.load.rawbuffer.f32(
      target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 0, i32 0)
  %data = extractvalue {float, i1} %load, 0
  ret float %data
}

; CHECK: Function structbuf : 0x00000010
define float @structbuf() "hlsl.export" {
  %buffer = call target("dx.RawBuffer", float, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %load = call {float, i1} @llvm.dx.resource.load.rawbuffer.f32(
      target("dx.RawBuffer", float, 0, 0, 0) %buffer, i32 0, i32 0)
  %data = extractvalue {float, i1} %load, 0
  ret float %data
}

; CHECK: Function typedbuf : 0x00000000
define float @typedbuf(<4 x float> %val) "hlsl.export" {
  %buffer = call target("dx.TypedBuffer", float, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %load = call {float, i1} @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", float, 0, 0, 0) %buffer, i32 0)
  %data = extractvalue {float, i1} %load, 0
  ret float %data
}
