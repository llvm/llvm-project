; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

; This test checks to ensure the behavior of the DXIL shader flag analysis
; for the flag ResMayNotAlias is correct when the DXIL Version is < 1.7. The
; ResMayNotAlias flag (0x20000000) should not be set at all.

target triple = "dxil-pc-shadermodel6.6-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00000010

; CHECK: Note: extra DXIL module flags:
; CHECK:       Raw and structured buffers
; CHECK-NOT:   Any UAV may not alias any other UAV
;

; CHECK: Function loadUAV : 0x00000000
define float @loadUAV() #0 {
  %res = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %load = call {float, i1} @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", float, 1, 0, 0) %res, i32 0)
  %val = extractvalue {float, i1} %load, 0
  ret float %val
}

; CHECK: Function loadSRV : 0x00000010
define float @loadSRV() #0 {
  %res = tail call target("dx.RawBuffer", float, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %load = call {float, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.RawBuffer", float, 0, 0) %res, i32 0, i32 0)
  %val = extractvalue { float, i1 } %load, 0
  ret float %val
}

!dx.valver = !{!0}
!0 = !{i32 1, i32 8}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
